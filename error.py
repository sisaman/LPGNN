import os
import datetime
from argparse import ArgumentParser
from itertools import product

import pandas as pd
import torch

from datasets import supported_datasets, load_dataset
from models import KProp
from mechanisms import supported_mechanisms
from transforms import Privatize
from utils import colored_text, print_args, seed_everything, measure_runtime


class GConv(KProp):
    def __init__(self, aggregator):
        super().__init__(
            in_channels=1, out_channels=1, step=1,
            aggregator=aggregator, add_self_loops=True, cached=False
        )

    def forward(self, x, adj_t):
        return self.neighborhood_aggregation(x, adj_t)


class ErrorEstimation:
    def __init__(self, method, eps, aggr, device='cuda'):
        self.method = method
        self.eps = eps
        device = 'cpu' if not torch.cuda.is_available() else device
        self.model = GConv(aggregator=aggr).to(device)

    def calculate_error(self, data_priv, norm):
        hn = self.model(data_priv.x_raw, data_priv.adj_t)
        hn_hat = self.model(data_priv.x, data_priv.adj_t)
        diff = hn - hn_hat
        errors = torch.norm(diff, p=norm, dim=1) / data_priv.num_features
        return errors

    def estimate(self, data):
        privatize = Privatize(method=self.method, epsilon=self.eps)
        data = privatize(data)
        errors = self.calculate_error(data, norm=1)
        return {'error_mean': errors.mean().item(), 'error_std': errors.std().item()}


@measure_runtime
@torch.no_grad()
def run(args):
    results = []
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name=dataset_name, data_range=(0, 1), sparse=True).to(args.device)
        configs = product(args.methods, args.aggs, args.epsilons)
        for method, agg, eps in configs:
            experiment_name = ', '.join([f'dataset:{dataset.name}', f'method:{method}', f'eps:{eps}', f'agg:{agg}'])
            print(experiment_name)
            task = ErrorEstimation(method=method, eps=eps, aggr=agg, device=args.device)
            result = task.estimate(dataset)
            result.update(dataset_name=dataset.name, method=method, aggregator=agg, epsilon=eps)
            results.append(result)

    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(args.output_dir, f'{str(datetime.datetime.now())}.csv'), index=False)


def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-d', '--datasets', nargs='+', choices=supported_datasets, default=list(supported_datasets))
    parser.add_argument('-m', '--methods', nargs='+', choices=supported_mechanisms, default=list(supported_mechanisms))
    parser.add_argument('-e', '--epsilons', nargs='+', type=float, dest='epsilons', required=True)
    parser.add_argument('-a', '--aggs', nargs='*', type=str, default=['gcn'])
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default='./output/error')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # check if eps > 0 for LDP methods
    if min(args.epsilons) <= 0:
        parser.error('LDP methods require eps > 0.')

    if not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.device = 'cpu'

    if args.seed:
        seed_everything(args.seed)

    print_args(args)
    run(args)


if __name__ == '__main__':
    main()
