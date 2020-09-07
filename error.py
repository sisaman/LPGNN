import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
import pandas as pd
from datasets import get_available_datasets, GraphDataModule
from transforms import Privatize
from utils import TermColors
from privacy import get_available_mechanisms


class ErrorEstimation:
    def __init__(self, data, logger, max_degree_quantile=0.99, device='cuda'):
        self.data = data
        alpha = data.x_raw.min(dim=0)[0]
        beta = data.x_raw.max(dim=0)[0]
        self.sensitivity = beta - alpha
        self.logger = logger
        self.max_degree_quantile = max_degree_quantile
        device = 'cpu' if not torch.cuda.is_available() else device

        self.model = GCNConv(data.num_features, data.num_features, cached=True).to(device)
        self.model.weight.data.copy_(torch.eye(data.num_features))  # identity transformation
        self.gc = self.model(data.x_raw, data.edge_index)

    def run(self):
        gc_hat = self.model(self.data.x, self.data.edge_index)
        diff = (self.gc - gc_hat) / self.sensitivity
        errors = torch.norm(diff, p=1, dim=1) / self.data.num_features
        degrees = degree(self.data.edge_index[0], self.data.num_nodes)

        df = pd.DataFrame({'degree': degrees.cpu(), 'error': errors.cpu()})
        df = df[df['degree'] < df['degree'].quantile(q=self.max_degree_quantile)]
        values = df.groupby('degree').agg(['mean', 'std']).fillna(0).reset_index().values

        for deg, mae, std in values:
            self.logger.log_metrics(metrics={'mae': mae, 'std': std}, step=deg)


def error_estimation(dataset, method, eps, repeats, output_dir, device):
    for run in range(repeats):
        params = {
            'task': 'error',
            'dataset': dataset.name,
            'method': method,
            'eps': eps,
            'run': run
        }

        params_str = ' | '.join([f'{key}={val}' for key, val in params.items()])
        print(TermColors.FG.green + params_str + TermColors.reset)

        output_dir = os.path.join(output_dir, 'error', dataset.name, method, str(eps))
        logger = TensorBoardLogger(save_dir=output_dir, name=None)

        privatize = Privatize(method=method, eps=eps)
        dataset.add_transform(privatize)
        ErrorEstimation(data=dataset[0], logger=logger, device=device).run()


@torch.no_grad()
def batch_error_estimation(args):
    for dataset_name in args.datasets:
        dataset = GraphDataModule(name=dataset_name, normalize=(0, 1), device=args.device)
        for method in args.methods:
            for eps in args.epsilons:
                error_estimation(
                    dataset=dataset,
                    method=method,
                    eps=eps,
                    repeats=args.repeats,
                    output_dir=args.output_dir,
                    device=args.device
                )


def main():
    seed_everything(12345)

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-d', '--datasets', nargs='+', choices=get_available_datasets(), required=True)
    parser.add_argument('-m', '--methods',  nargs='+', choices=get_available_mechanisms(), required=True)
    parser.add_argument('-e', '--eps',      nargs='+', type=float, dest='epsilons', required=True)
    parser.add_argument('-r', '--repeats',      type=int, default=1)
    parser.add_argument('-o', '--output-dir',   type=str, default='./results')
    parser.add_argument('--device',             type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # check if eps > 0 for LDP methods
    if min(args.epsilons) <= 0:
        parser.error('LDP methods require eps > 0.')

    print(args)
    batch_error_estimation(args)


if __name__ == '__main__':
    main()
