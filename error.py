import os
from argparse import ArgumentParser
from itertools import product

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

from datasets import available_datasets, GraphDataModule
from models import KProp
from privacy import available_mechanisms
from transforms import Privatize
from utils import colored_print


class GConv(KProp):
    def __init__(self, aggregator):
        super().__init__(
            in_channels=1, out_channels=1, K=1,
            aggregator=aggregator, add_self_loops=True, cached=False
        )

    def forward(self, x, adj_t):
        return self.neighborhood_aggregation(x, adj_t)


class ErrorEstimation:
    available_tasks = ['eps', 'deg']

    def __init__(self, task, method, eps, aggr, logger, device='cuda'):
        self.task = task
        self.method = method
        self.eps = eps
        self.logger = logger
        device = 'cpu' if not torch.cuda.is_available() else device
        self.model = GConv(aggregator=aggr).to(device)

    def run(self, data):
        if self.task == 'eps':
            self.error_eps(data)
        elif self.task == 'deg':
            self.error_degree(data)
        else:
            raise ValueError('task not supported')

    def calculate_error(self, data_priv, norm):
        hn = self.model(data_priv.x_raw, data_priv.adj_t)
        hn_hat = self.model(data_priv.x, data_priv.adj_t)
        diff = hn - hn_hat
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
        degrees = data.adj_t.sum(dim=0)
        df = pd.DataFrame({'degree': degrees.cpu(), 'error': errors.cpu()})
        df = df[df['degree'] < df['degree'].quantile(q=0.99)]
        df.apply(lambda row: self.logger.log_metrics(metrics={'error': row['error'], 'degree': row['degree']}), axis=1)


def error_estimation(task, dataset, method, eps, aggr, repeats, output_dir, device):
    experiment_dir = os.path.join(
        f'task:{task}',
        f'dataset:{dataset.name}',
        f'method:{method}',
        f'eps:{eps}',
        f'agg:{aggr}',
    )
    colored_print(experiment_dir, color='green')

    for run in range(repeats):
        output_dir = os.path.join(output_dir, experiment_dir)
        logger = CSVLogger(save_dir=output_dir, name=None)
        task = ErrorEstimation(task=task, method=method, eps=eps, aggr=aggr, logger=logger, device=device)
        task.run(dataset[0])
        logger.save()


@torch.no_grad()
def batch_error_estimation(args):
    for dataset_name in args.datasets:
        dataset = GraphDataModule(name=dataset_name, feature_range=(0, 1), sparse=True, device=args.device)
        configs = product(args.methods, args.epsilons, args.aggs)
        for method, eps, agg in configs:
            error_estimation(
                task=args.task,
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
    parser.add_argument('-t', '--task', type=str, choices=ErrorEstimation.available_tasks, required=True)
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

    print(args)
    batch_error_estimation(args)


if __name__ == '__main__':
    main()
