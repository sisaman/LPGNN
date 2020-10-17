import logging
import os
import sys
import time
from argparse import ArgumentParser
from itertools import product

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from datasets import available_datasets, load_dataset
from models import NodeClassifier
from privacy import available_mechanisms
from transforms import Privatize, LabelRate
from utils import ProgressBar, colored_text, print_args


def train_and_test(dataset, label_rate, method, eps, K, aggregator, args, experiment_dir):
    # define model
    model = NodeClassifier(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        aggregator=aggregator,
        K=K,
        **vars(args)
    )

    # define trainer
    trainer = Trainer.from_argparse_args(
        args=args,
        precision=32,
        gpus=int(args.device == 'cuda' and torch.cuda.is_available()),
        max_epochs=500,
        callbacks=[ProgressBar(process_position=1, refresh_rate=50)],
        checkpoint_callback=ModelCheckpoint(monitor='val_loss', filepath=os.path.join('checkpoints', experiment_dir)),
        weights_summary=None,
        deterministic=True,
        logger=False,
        num_sanity_val_steps=0
        # logger=TensorBoardLogger(save_dir=os.path.join(output_dir, experiment_dir), name=None),
    )

    # apply transforms
    dataset = LabelRate(rate=label_rate)(dataset)
    dataset = Privatize(method=method, eps=eps)(dataset)

    # train and test
    dataloader = {dataset}
    trainer.fit(model=model, train_dataloader=dataloader, val_dataloaders=dataloader)
    result = trainer.test(test_dataloaders=dataloader, ckpt_path='best', verbose=False)
    return result[0]['test_acc']


def batch_train_and_test(args):
    if args.random_seed is not None:
        seed_everything(args.random_seed)

    dataset = load_dataset(name=args.dataset, feature_range=(0, 1), sparse=True,
                           device=args.device, random_state=args.random_seed)

    non_priv_methods = {'raw', 'rnd'} & set(args.methods)
    priv_methods = set(args.methods) - non_priv_methods
    configs = list(product(non_priv_methods, args.label_rates, [0.0], args.steps, args.aggs))
    configs += list(product(priv_methods, args.label_rates, args.epsilons, args.steps, args.aggs))

    for method, lr, eps, k, aggr in configs:
        experiment_dir = os.path.join(
            f'task:train', f'dataset:{dataset.name}', f'labelrate:{lr}', f'method:{method}',
            f'eps:{eps}', f'step:{k}', f'agg:{aggr}', f'selfloops:{args.self_loops}'
        )

        results = []
        run_desc = colored_text(experiment_dir.replace('/', ', '), color='green')
        progbar = tqdm(range(args.repeats), desc=run_desc, file=sys.stdout)
        for _ in progbar:
            result = train_and_test(
                dataset=dataset, label_rate=lr, method=method,
                eps=eps, K=k, aggregator=aggr, args=args, experiment_dir=experiment_dir
            )

            results.append(result)
            progbar.set_postfix({'last_test_acc': results[-1], 'avg_test_acc': np.mean(results)})

        # save results
        save_dir = os.path.join(args.output_dir, experiment_dir)
        os.makedirs(save_dir, exist_ok=True)
        df_results = pd.DataFrame(results, columns=['test_acc']).rename_axis('version').reset_index()
        df_results.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)


def main():
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=available_datasets(), required=True)
    parser.add_argument('-l', '--label-rate', type=float, default=1)
    parser.add_argument('-m', '--method', choices=available_mechanisms() + ['raw', 'rnd'], default='raw')
    parser.add_argument('-e', '--epsilon', type=float, default=0)
    parser.add_argument('-k', '--step', type=int, default=1)
    parser.add_argument('-a', '--aggregator', type=str, default='gcn')
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('-s', '--random-seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser = NodeClassifier.add_module_specific_args(parser)
    args = parser.parse_args()

    if len(set(args.methods) & set(available_mechanisms())) > 0:
        if min(args.epsilons) <= 0:
            parser.error('LDP methods require eps > 0.')

    print_args(args)
    start = time.time()
    batch_train_and_test(args)
    end = time.time()
    print('\nTotal time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
