import logging
import os
import time
from tabulate import tabulate
from argparse import ArgumentParser
from itertools import product
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm
from datasets import available_datasets, GraphDataModule
from models import NodeClassifier
from privacy import available_mechanisms
from transforms import Privatize, LabelRate
from utils import ProgressBar, colored_text


def train_and_test(dataset, label_rate, method, eps, K, aggregator, args, repeats, output_dir):
    experiment_dir = os.path.join(
        f'task:train',
        f'dataset:{dataset.name}',
        f'labelrate:{label_rate}',
        f'method:{method}',
        f'eps:{eps}',
        f'step:{K}',
        f'agg:{aggregator}',
        f'selfloops:{args.self_loops}'
    )

    results = []
    progbar = tqdm(range(repeats), desc=colored_text(experiment_dir.replace('/', ', '), color='green'))
    for run in progbar:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filepath=os.path.join('checkpoints', experiment_dir)
        )

        model = NodeClassifier(
            aggregator=aggregator,
            K=K,
            **vars(args)
        )

        trainer = Trainer.from_argparse_args(
            args=args,
            precision=32,
            gpus=int(args.device == 'cuda' and torch.cuda.is_available()),
            max_epochs=500,
            callbacks=[ProgressBar(process_position=1, refresh_rate=50)],
            checkpoint_callback=checkpoint_callback,
            weights_summary=None,
            deterministic=True,
            logger=False,
            num_sanity_val_steps=int(run == 0)
            # logger=TensorBoardLogger(save_dir=os.path.join(output_dir, experiment_dir), name=None),
        )

        dataset.add_transform(Privatize(method=method, eps=eps))
        dataset.add_transform(LabelRate(rate=label_rate))
        trainer.fit(model=model, datamodule=dataset)
        result = trainer.test(datamodule=dataset, ckpt_path='best', verbose=False)
        results.append(result[0]['test_acc'])

        progbar.set_postfix({'last_test_acc': results[-1], 'avg_test_acc': np.mean(results)})

    # save results
    save_dir = os.path.join(output_dir, experiment_dir)
    os.makedirs(save_dir, exist_ok=True)
    df_results = pd.DataFrame(results, columns=['test_acc']).rename_axis('version').reset_index()
    df_results.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)


def batch_train_and_test(args):
    dataset = GraphDataModule(name=args.dataset, feature_range=(0, 1), sparse=True, device=args.device)
    non_priv_methods = {'raw', 'rnd'} & set(args.methods)
    priv_methods = set(args.methods) - non_priv_methods
    configs = list(product(non_priv_methods, args.label_rates, [0.0], args.steps, args.aggs))
    configs += list(product(priv_methods, args.label_rates, args.epsilons, args.steps, args.aggs))

    for method, lr, eps, steps, aggr in configs:
        train_and_test(
            dataset=dataset,
            label_rate=lr,
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
    parser.add_argument('-l', '--label-rates', type=float, nargs='*', default=[1.0])
    parser.add_argument('-m', '--methods', nargs='+', choices=available_mechanisms() + ['raw', 'rnd'], required=True)
    parser.add_argument('-e', '--epsilons', nargs='*', type=float, default=[1])
    parser.add_argument('-k', '--steps', nargs='*', type=int, default=[1])
    parser.add_argument('-a', '--aggs', nargs='*', type=str, default=['gcn'])
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser = NodeClassifier.add_module_specific_args(parser)
    args = parser.parse_args()

    if len(set(args.methods) & set(available_mechanisms())) > 0:
        if min(args.epsilons) <= 0:
            parser.error('LDP methods require eps > 0.')

    df_args = pd.DataFrame(vars(args), index=['']).T
    print(tabulate(df_args, tablefmt='fancy_grid'), '\n')
    start = time.time()
    batch_train_and_test(args)
    end = time.time()
    print('\nTotal time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
