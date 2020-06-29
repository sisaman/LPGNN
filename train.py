import logging

logging.captureWarnings(True)

from argparse import ArgumentParser

import time
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.data import DataLoader
from datasets import load_dataset, get_available_datasets
from privacy import privatize, get_available_mechanisms
from models import NodeClassifier, LinkPredictor
from utils import TrainOnlyProgressBar, PandasLogger, TermColors


class GraphTask:
    task_models = {
        'node': NodeClassifier,
        'link': LinkPredictor
    }

    @staticmethod
    def get_available_tasks():
        return list(GraphTask.task_models.keys())

    @staticmethod
    def add_task_specific_args(task_name, parent_parser):
        return GraphTask.task_models[task_name].add_model_specific_args(parent_parser)

    def __init__(self, logger, hparams):
        self.hparams = hparams
        self.trainer = Trainer.from_argparse_args(
            args=self.hparams,
            gpus=int(hparams.device == 'cuda' and torch.cuda.is_available()),
            callbacks=[TrainOnlyProgressBar()],
            checkpoint_callback=False,
            logger=logger,
            row_log_interval=1000,
            log_save_interval=1000,
            weights_summary=None,
            deterministic=True,
            progress_bar_refresh_rate=5,
            early_stop_callback=EarlyStopping(min_delta=self.hparams.min_delta, patience=self.hparams.patience),
        )

    def train(self, data):
        params = vars(self.hparams)
        params['input_dim'] = data.num_features
        if self.hparams.task == 'node':
            params['num_classes'] = data.num_classes
        model = self.task_models[self.hparams.task](**params)
        dataloader = DataLoader([data])
        self.trainer.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)

    def test(self, data):
        dataloader = DataLoader([data])
        self.trainer.test(test_dataloaders=dataloader, ckpt_path=None)


def train_and_test(task, data, method, eps, hparams, logger, repeats):
    for run in range(repeats):
        params = {
            'task': task,
            'dataset': data.name,
            'method': method,
            'eps': eps,
            'run': run
        }

        params_str = ' | '.join([f'{key}={val}' for key, val in params.items()])
        print(TermColors.FG.green + params_str + TermColors.reset)
        logger.log_params(params)

        data_priv = privatize(data, method=method, eps=eps)
        t = GraphTask(logger, hparams)
        t.train(data_priv)
        t.test(data_priv)


def batch_train_and_test(args):
    data = load_dataset(args.dataset, split_edges=(args.task == 'link'), device=args.device)
    for method in args.methods:
        experiment_name = f'{args.task}_{args.dataset}_{method}'
        with PandasLogger(
                output_dir=args.output_dir,
                experiment_name=experiment_name,
                write_mode='replace'
        ) as logger:
            for eps in args.eps_list:
                train_and_test(
                    task=args.task,
                    data=data,
                    method=method,
                    eps=eps,
                    hparams=args,
                    repeats=args.repeats,
                    logger=logger,
                )


def main():
    seed_everything(12345)
    parser = ArgumentParser()

    parser.add_argument('-t', '--task', type=str, choices=GraphTask.get_available_tasks(), required=True,
                        help='The graph learning task. Either "node" for node classification, '
                             'or "link" for link prediction.'
                        )
    parser.add_argument('-d', '--dataset', type=str, choices=get_available_datasets(), required=True,
                        help='The dataset to train on. One of "citeseer", "cora", "elliptic", "flickr", and "twitch".'
                        )
    parser.add_argument('-m', '--methods', nargs='+', choices=get_available_mechanisms() + ['raw'], required=True,
                        help='The list of mechanisms to perturb node features. '
                             'Can choose "raw" to use original features, or "pgc" for Private Graph Convolution, '
                             '"pm" for Piecewise Mechanism, and "lm" for Laplace Mechanism, '
                             'as local differentially private algorithms.'
                        )
    parser.add_argument('-e', '--eps', nargs='*', type=float, dest='eps_list', default=[0],
                        help='The list of epsilon values for LDP mechanisms. The values must be greater than zero. '
                             'The "raw" method does not support this options.'
                        )
    parser.add_argument('-r', '--repeats', type=int, default=10,
                        help='The number of repeating the experiment. Defaults to 10.'
                        )
    parser.add_argument('-o', '--output-dir', type=str, default='./results',
                        help='The path to store the results. Defaults to "./results".'
                        )
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='The device used for the training. Either "cpu" or "cuda". Defaults to "cuda".'
                        )

    # add args based on the task
    temp_args, _ = parser.parse_known_args()
    parser = GraphTask.add_task_specific_args(task_name=temp_args.task, parent_parser=parser)

    # check if eps > 0 for LDP methods
    if len(set(temp_args.methods).intersection(get_available_mechanisms())) > 0:
        for eps in temp_args.eps_list:
            if eps <= 0:
                parser.error('LDP methods require eps > 0.')

    args = parser.parse_args()
    print(args)

    start = time.time()
    batch_train_and_test(args)
    end = time.time()
    print('Total time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
