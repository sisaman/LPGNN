import logging
import os
import sys
import time
from argparse import ArgumentParser
from itertools import product

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import available_datasets, load_dataset
from models import NodeClassifier
from transforms import FeatureTransform, LabelRate
from utils import colored_text, print_args, seed_everything


class Trainer:
    def __init__(self, max_epochs=100, device='cuda', checkpoint_dir='checkpoints'):
        self.max_epochs = max_epochs
        self.device = 'cpu' if not torch.cuda.is_available() else device
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, 'best_weights.pt')

    def fit(self, model, data):
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = model.configure_optimizers()

        best_val_loss = float('inf')

        epoch_progbar = tqdm(range(1, self.max_epochs + 1), desc='Epoch: ', leave=False, position=1, file=sys.stdout)
        for _ in epoch_progbar:
            loss, metric = self.train(model, data, optimizer)
            val_metrics = self.validation(model, data)
            val_loss = val_metrics['val_loss']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.checkpoint_path)

            # display metrics on progress bar
            postfix = {'train_loss': loss.item(), **metric, **val_metrics}
            epoch_progbar.set_postfix(postfix)

        return model

    @torch.no_grad()
    def test(self, model, data):
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()
        return model.test_step(data)

    @torch.no_grad()
    def validation(self, model, data):
        model.eval()
        return model.validation_step(data)

    @staticmethod
    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model.training_step(data)
        loss.backward()
        optimizer.step()
        return loss, metrics


def train_and_test(dataset, label_rate, eps, K, checkpoint_path, args):
    # define model
    model = NodeClassifier(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        K=K,
        **vars(args)
    )

    # apply transforms
    dataset = LabelRate(rate=label_rate)(dataset)
    dataset = FeatureTransform(method=args.method, eps=eps)(dataset)

    trainer = Trainer(max_epochs=500, device=args.device, checkpoint_dir=checkpoint_path)
    model = trainer.fit(model, dataset)
    result = trainer.test(model, dataset)

    return result['test_acc']


def batch_train_and_test(args):
    dataset = load_dataset(name=args.dataset, feature_range=(0, 1), sparse=True, device=args.device)
    configs = list(product(args.label_rates, args.epsilons, args.steps))

    for lr, eps, k in configs:
        experiment_dir = os.path.join(
            f'task:train', f'dataset:{args.dataset}', f'labelrate:{lr}', f'method:{args.method}',
            f'eps:{eps}', f'step:{k}', f'agg:{args.aggregator}', f'selfloops:{args.self_loops}'
        )

        results = []
        run_desc = colored_text(experiment_dir.replace('/', ', '), color='green')
        progbar = tqdm(range(args.repeats), desc=run_desc, file=sys.stdout)
        for run in progbar:
            result = train_and_test(
                dataset=dataset, label_rate=lr,
                eps=eps, K=k, args=args,
                checkpoint_path=os.path.join('checkpoints', experiment_dir, str(run))
            )

            results.append(result)
            progbar.set_postfix({'last_test_acc': results[-1], 'avg_test_acc': np.mean(results)})

        # save results
        save_dir = os.path.join(args.output_dir, experiment_dir)
        os.makedirs(save_dir, exist_ok=True)
        df_results = pd.DataFrame(results, columns=['test_acc']).rename_axis('version').reset_index()
        df_results.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)


def main():
    seed_everything(12345)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=available_datasets(), required=True)
    parser.add_argument('-m', '--method', type=str, choices=FeatureTransform.available_methods(), required=True)
    parser.add_argument('-e', '--epsilons', nargs='*', type=float, default=[0.0])
    parser.add_argument('-k', '--steps', nargs='*', type=int, default=[1])
    parser.add_argument('-l', '--label-rates', nargs='*', type=float, default=[1.0])
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser = NodeClassifier.add_module_specific_args(parser)
    args = parser.parse_args()

    if args.method in FeatureTransform.private_methods and min(args.epsilons) <= 0:
        parser.error('LDP method requires eps > 0.')

    print_args(args)
    start = time.time()
    batch_train_and_test(args)
    end = time.time()
    print('\nTotal time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
