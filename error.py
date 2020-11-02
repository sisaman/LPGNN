import os
from argparse import ArgumentParser
from itertools import product

import pandas as pd
import torch
from tqdm.auto import tqdm

from datasets import available_datasets, load_dataset
from models import KProp
from privacy import available_mechanisms
from transforms import FeatureTransform
from utils import colored_text, print_args, seed_everything


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

    def run(self, data):
        privatize = FeatureTransform(method=self.method, eps=self.eps)
        data = privatize(data)
        errors = self.calculate_error(data, norm=1)
        return errors.mean().item(), errors.std().item()


def error_estimation(dataset, method, eps, aggr, repeats, output_dir, device):
    experiment_dir = os.path.join(
        f'task:error',
        f'dataset:{dataset.name}',
        f'method:{method}',
        f'eps:{eps}',
        f'agg:{aggr}',
    )

    results = []
    progbar = tqdm(range(repeats), desc=colored_text(experiment_dir.replace('/', ', '), color='green'))
    for _ in progbar:
        task = ErrorEstimation(method=method, eps=eps, aggr=aggr, device=device)
        result = task.run(dataset)
        results.append(result)

    # save results
    save_dir = os.path.join(output_dir, experiment_dir)
    os.makedirs(save_dir, exist_ok=True)
    df_results = pd.DataFrame(results, columns=['error', 'std']).rename_axis('version').reset_index()
    df_results.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)


@torch.no_grad()
def batch_error_estimation(args):
    for dataset_name in args.datasets:
        dataset = load_dataset(name=dataset_name, feature_range=(0, 1), sparse=True).to(args.device)
        configs = product(args.methods, args.epsilons, args.aggs)
        for method, eps, agg in configs:
            error_estimation(
                dataset=dataset,
                method=method,
                eps=eps,
                aggr=agg,
                repeats=args.repeats,
                output_dir=args.output_dir,
                device=args.device,
            )


def main():
    seed_everything(12345)

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-d', '--datasets', nargs='+', choices=available_datasets(), default=available_datasets())
    parser.add_argument('-m', '--methods', nargs='+', choices=available_mechanisms(), default=available_mechanisms())
    parser.add_argument('-e', '--epsilons', nargs='+', type=float, dest='epsilons', required=True)
    parser.add_argument('-a', '--aggs', nargs='*', type=str, default=['gcn'])
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # check if eps > 0 for LDP methods
    if min(args.epsilons) <= 0:
        parser.error('LDP methods require eps > 0.')

    print_args(args)
    batch_error_estimation(args)


if __name__ == '__main__':
    main()
