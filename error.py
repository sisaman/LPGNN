from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
import pandas as pd
from datasets import get_available_datasets, GraphDataset
from utils import TermColors
from privacy import get_available_mechanisms, Privatize


class ErrorEstimation:
    def __init__(self, data, max_degree_quantile=0.99, device='cuda'):
        self.data = data
        alpha = data.x_raw.min(dim=0)[0]
        beta = data.x_raw.max(dim=0)[0]
        self.delta = beta - alpha
        self.max_degree_quantile = max_degree_quantile

        self.model = GCNConv(data.num_features, data.num_features, cached=True)
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.weight.data.copy_(torch.eye(data.num_features))  # identity transformation
        self.gc = self.model(data.x_raw, data.edge_index)

    def run(self, logger):
        # calculate error
        gc_hat = self.model(self.data.x, self.data.edge_index)
        diff = (self.gc - gc_hat) / self.delta
        diff[:, (self.delta == 0)] = 0  # avoid division by zero
        errors = torch.norm(diff, p=1, dim=1) / self.data.num_features

        # obtain node degrees
        row, col = self.data.edge_index
        degrees = degree(row, self.data.num_nodes)

        df = pd.DataFrame({'degree': degrees.cpu(), 'error': errors.cpu()})
        df = df[df['degree'] < df['degree'].quantile(q=self.max_degree_quantile)]
        values = df.groupby('degree').agg(['mean', 'std']).fillna(0).reset_index().values

        for deg, mae, std in values:
            logger.log_metrics(metrics={'mae': mae, 'std': std}, step=deg)


def error_estimation(dataset, method, eps, repeats, save_dir, device):
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

        experiment_name = f'error_{dataset.name}_{method}_{eps}'
        logger = TensorBoardLogger(save_dir=save_dir, name=experiment_name, version=run)

        data = Privatize(method=method, eps=eps)(dataset.get_data())
        ErrorEstimation(data=data, device=device).run(logger)


@torch.no_grad()
def batch_error_estimation(args):
    for dataset_name in args.datasets:
        dataset = GraphDataset(dataset_name=dataset_name, device=args.device)
        for method in args.methods:
            for eps in args.eps_list:
                error_estimation(
                    dataset=dataset,
                    method=method,
                    eps=eps,
                    repeats=args.repeats,
                    save_dir=args.output_dir,
                    device=args.device
                )


def main():
    seed_everything(12345)

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-d', '--datasets', nargs='+', choices=get_available_datasets(), required=True)
    parser.add_argument('-m', '--methods',  nargs='+', choices=get_available_mechanisms(), required=True)
    parser.add_argument('-e', '--eps',      nargs='+', type=float, dest='eps_list', required=True)
    parser.add_argument('-r', '--repeats',      type=int, default=1)
    parser.add_argument('-o', '--output-dir',   type=str, default='./results')
    parser.add_argument('--device',             type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    print(args)

    batch_error_estimation(args)


if __name__ == '__main__':
    main()
