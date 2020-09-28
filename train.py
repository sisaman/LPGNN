import logging
import os
import time
from argparse import ArgumentParser
from itertools import product

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import available_datasets, GraphDataModule
from models import NodeClassifier
from privacy import available_mechanisms
from transforms import Privatize
from utils import TermColors


def train_and_test(dataset, method, eps, K, aggregator, args, repeats, output_dir):
    experiment_dir = os.path.join(
        'task:node',
        f'dataset:{dataset.name}',
        f'method:{method}',
        f'eps:{eps}',
        f'step:{K}',
        f'agg:{aggregator}',
        f'loops:{args.self_loops}'
    )
    print(TermColors.FG.green + experiment_dir + TermColors.reset)

    for run in range(repeats):
        print(TermColors.FG.lightblue + f'Run {run}:' + TermColors.reset)
        logger = TensorBoardLogger(save_dir=os.path.join(output_dir, experiment_dir), name=None)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', filepath=os.path.join('checkpoints', experiment_dir))

        log_learning_curve = run == 0 and (method == 'raw' or method == 'mbm')
        model = NodeClassifier(aggregator=aggregator, K=K, log_learning_curve=log_learning_curve, **vars(args))

        trainer = Trainer.from_argparse_args(
            args=args,
            precision=32,
            gpus=int(args.device == 'cuda' and torch.cuda.is_available()),
            max_epochs=500,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            log_save_interval=500,
            weights_summary=None,
            deterministic=True,
            progress_bar_refresh_rate=10,
        )

        privatize = Privatize(method=method, eps=eps)
        dataset.add_transform(privatize)
        trainer.fit(model=model, datamodule=dataset)
        trainer.test(datamodule=dataset, ckpt_path='best', verbose=True)


def batch_train_and_test(args):
    dataset = GraphDataModule(name=args.dataset, normalize=(0, 1), sparse=True, device=args.device)
    non_priv_methods = {'raw', 'cst'} & set(args.methods)
    priv_methods = set(args.methods) - non_priv_methods

    configs = []
    for method in non_priv_methods:
        configs += list(product(method, [0.0], args.steps, args.aggs))  # bind non-private methods with eps=0

    for method in priv_methods:
        configs += list(product(method, args.epsilons, args.steps, args.aggs))

    for method, eps, steps, aggr in configs:
        train_and_test(
            dataset=dataset,
            method=method,
            eps=eps,
            K=steps,
            aggregator=aggr,
            args=args,
            repeats=args.repeats,
            output_dir=args.output_dir
        )


def main():
    seed_everything(12345)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=available_datasets(), required=True)
    parser.add_argument('-m', '--methods', nargs='+', choices=available_mechanisms() + ['raw', 'cst'], required=True)
    parser.add_argument('-e', '--epsilons', nargs='*', type=float, default=[1])
    parser.add_argument('-k', '--steps', nargs='*', type=int, default=[1])
    parser.add_argument('-a', '--aggs', nargs='*', type=str, default=['gcn'])
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser = NodeClassifier.add_module_specific_args(parser)
    args = parser.parse_args()

    # check if eps > 0 for LDP methods
    if len(set(args.methods) & set(available_mechanisms())) > 0:
        if min(args.epsilons) <= 0:
            parser.error('LDP methods require eps > 0.')

    print(args)
    start = time.time()
    batch_train_and_test(args)
    end = time.time()
    print('\nTotal time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
