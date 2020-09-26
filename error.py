import os
from argparse import ArgumentParser
from itertools import product

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.utils import degree
from tqdm import tqdm

from datasets import available_datasets, GraphDataModule
from models import KProp
from privacy import available_mechanisms
from transforms import Privatize
from utils import TermColors


class KPropError(KProp):
    def __init__(self, K, aggregator):
        super().__init__(in_channels=1, out_channels=1, K=K, aggregator=aggregator, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        return self.neighborhood_aggregation(x, edge_index, edge_weight)


class ErrorEstimation:
    available_tasks = ['eps', 'deg', 'dim']

    def __init__(self, task, method, eps, k, agg, logger, device='cuda'):
        self.task = task
        self.method = method
        self.eps = eps
        self.logger = logger
        device = 'cpu' if not torch.cuda.is_available() else device
        self.model = KPropError(K=k, aggregator=agg).to(device)
        self.cache = None

    def run(self, data):
        if self.task == 'eps':
            self.error_eps(data)
        elif self.task == 'deg':
            self.error_degree(data)
        elif self.task == 'dim':
            self.error_dimension(data)
        else:
            raise ValueError('mode not supported')

    def calculate_error(self, data_priv, norm, cached=False):
        if self.cache is None or not cached:
            alpha = data_priv.x_raw.min(dim=0)[0]
            beta = data_priv.x_raw.max(dim=0)[0]
            hn = self.model(data_priv.x_raw, data_priv.edge_index)
            self.cache = alpha, beta, hn

        alpha, beta, hn = self.cache
        hn_hat = self.model(data_priv.x, data_priv.edge_index)
        diff = (hn - hn_hat) / (beta - alpha)
        errors = torch.norm(diff, p=norm, dim=1) / data_priv.num_features
        return errors

    def error_eps(self, data):
        privatize = Privatize(method=self.method, eps=self.eps)
        data = privatize(data)
        errors = self.calculate_error(data, norm=1)
        self.logger.log_metrics(metrics={'error': errors.mean(), 'std': errors.std()})

    def error_degree(self, data):
        privatize = Privatize(method=self.method, eps=self.eps)
        data = privatize(data)
        errors = self.calculate_error(data, norm=1)
        degrees = degree(data.edge_index[0], data.num_nodes)
        df = pd.DataFrame({'degree': degrees.cpu(), 'error': errors.cpu()})
        df = df[df['degree'] < df['degree'].quantile(q=0.99)]
        df.apply(lambda row: self.logger.log_metrics(metrics={'error': row['error'], 'degree': row['degree']}), axis=1)

    def error_dimension(self, data):
        d = data.num_features
        for m in tqdm(range(1, d + 1)):
            data = Privatize(method='mbm', eps=self.eps, m=m)(data)
            errors = self.calculate_error(data, norm=1, cached=True)
            self.logger.log_metrics(metrics={'error': errors.mean(), 'std': errors.std(), 'm': m})


def error_estimation(task, dataset, method, eps, k, agg, repeats, output_dir, device):
    for run in range(repeats):
        params = {
            'task': task,
            'dataset': dataset.name,
            'method': method,
            'eps': eps,
            'steps': k,
            'agg': agg,
            'run': run
        }

        params_str = ' | '.join([f'{key}={val}' for key, val in params.items()])
        print(TermColors.FG.green + params_str + TermColors.reset)

        output_dir = os.path.join(output_dir, task, dataset.name, method, str(eps), str(k), agg)
        logger = CSVLogger(save_dir=output_dir, name=None)
        task = ErrorEstimation(task=task, method=method, eps=eps, k=k, agg=agg, logger=logger, device=device)
        task.run(dataset[0])
        logger.save()


@torch.no_grad()
def batch_error_estimation(args):
    for dataset_name in args.datasets:
        dataset = GraphDataModule(name=dataset_name, normalize=(0, 1), device=args.device)
        configs = product(args.methods, args.epsilons, args.steps, args.aggs)
        for method, eps, k, agg in configs:
            error_estimation(
                task=args.task,
                dataset=dataset,
                method=method,
                eps=eps,
                k=k,
                agg=agg,
                repeats=args.repeats,
                output_dir=args.output_dir,
                device=args.device,
            )


def main():
    seed_everything(12345)

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-t', '--task', type=str, choices=ErrorEstimation.available_tasks, required=True)
    parser.add_argument('-d', '--datasets', nargs='+', choices=available_datasets(), default=available_datasets())
    parser.add_argument('-m', '--methods', nargs='+', choices=available_mechanisms(), default=available_mechanisms())
    parser.add_argument('-e', '--epsilons', nargs='+', type=float, dest='epsilons', required=True)
    parser.add_argument('-k', '--steps', nargs='*', type=int, default=[1])
    parser.add_argument('-a', '--aggs', nargs='*', type=str, default=['gcn'])
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # check if eps > 0 for LDP methods
    if min(args.epsilons) <= 0:
        parser.error('LDP methods require eps > 0.')

    print(args)
    batch_error_estimation(args)


if __name__ == '__main__':
    main()
