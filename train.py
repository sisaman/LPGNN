import logging
import os
import sys
import time
from argparse import ArgumentParser

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


def train_and_test(args):
    if args.random_seed is not None:
        seed_everything(args.random_seed)

    dataset = load_dataset(name=args.dataset, feature_range=(0, 1), sparse=True,
                           device=args.device, random_state=args.random_seed)

    experiment_dir = os.path.join(
        f'task:train', f'dataset:{dataset.name}', f'labelrate:{args.label_rate}', f'method:{args.method}',
        f'eps:{args.epsilon}', f'step:{args.step}', f'agg:{args.aggregator}', f'selfloops:{args.self_loops}'
    )

    results = []
    run_desc = colored_text(experiment_dir.replace('/', ', '), color='green')
    progbar = tqdm(range(args.repeats), desc=run_desc, file=sys.stdout)
    for _ in progbar:
        model = NodeClassifier(
            input_dim=dataset.num_features,
            num_classes=dataset.num_classes,
            **vars(args)
        )

        # define trainer
        trainer = Trainer(
            precision=32,
            gpus=int(args.device == 'cuda' and torch.cuda.is_available()),
            max_epochs=500,
            callbacks=[ProgressBar(process_position=1, refresh_rate=50)],
            weights_summary=None,
            deterministic=True,
            logger=False,
            num_sanity_val_steps=0,
            checkpoint_callback=ModelCheckpoint(
                monitor='val_loss',
                filepath=os.path.join('checkpoints', experiment_dir)
            ),
            # logger=TensorBoardLogger(save_dir=os.path.join(output_dir, experiment_dir), name=None),
        )

        # apply transforms
        dataset = LabelRate(rate=args.label_rate)(dataset)
        dataset = Privatize(method=args.method, eps=args.epsilon)(dataset)

        # train and test
        dataloader = {dataset}
        trainer.fit(model=model, train_dataloader=dataloader, val_dataloaders=dataloader)
        result = trainer.test(test_dataloaders=dataloader, ckpt_path='best', verbose=False)
        results.append(result[0]['test_acc'])
        progbar.set_postfix({'last_test_acc': results[-1], 'avg_test_acc': np.mean(results)})

    # save results
    save_dir = os.path.join(args.output_dir, experiment_dir)
    os.makedirs(save_dir, exist_ok=True)
    df_results = pd.DataFrame(results, columns=['test_acc']).rename_axis('version').reset_index()
    df_results.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)


def main():
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)
    non_dp_methods = ['raw', 'rnd']

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=available_datasets(), required=True)
    parser.add_argument('-l', '--label-rate', type=float, default=1)
    parser.add_argument('-m', '--method', choices=available_mechanisms() + non_dp_methods, default='raw')
    parser.add_argument('-e', '--epsilon', type=float, default=0)
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('-s', '--random-seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser = NodeClassifier.add_module_specific_args(parser)
    args = parser.parse_args()

    if args.method in non_dp_methods:
        args.epsilon = 0
    elif args.epsilon <= 0:
        parser.error('LDP methods require eps > 0.')

    print_args(args)
    start = time.time()
    train_and_test(args)
    end = time.time()
    print('\nTotal time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
