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
from utils import colored_text, print_args, seed_everything, TensorBoardLogger


class Trainer:
    def __init__(self, max_epochs=100, device='cuda', checkpoint_dir='checkpoints', logger=None):
        self.max_epochs = max_epochs
        self.device = device
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, 'best_weights.pt')
        self.logger = logger

    def log(self, metrics, step=None):
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=step)

    def fit(self, model, data):
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = model.configure_optimizers()
        best_val_loss = float('inf')
        epoch_progbar = tqdm(range(1, self.max_epochs + 1), desc='Epoch: ', leave=False, position=1, file=sys.stdout)

        try:
            for epoch in epoch_progbar:
                train_metrics = self.__train(model, data, optimizer)
                train_metrics['train_loss'] = train_metrics['train_loss'].item()
                self.log(train_metrics, step=epoch)

                val_metrics = self.__validation(model, data)
                val_loss = val_metrics['val_loss']
                self.log(val_metrics, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), self.checkpoint_path)

                # display metrics on progress bar
                postfix = {**train_metrics, **val_metrics}
                epoch_progbar.set_postfix(postfix)
        except KeyboardInterrupt:
            pass

        if self.logger is not None:
            self.logger.save()
        return model

    @torch.no_grad()
    def test(self, model, data):
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()
        return model.test_step(data)

    @torch.no_grad()
    def __validation(self, model, data):
        model.eval()
        return model.validation_step(data)

    @staticmethod
    def __train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        metrics = model.training_step(data)
        loss = metrics['train_loss']
        loss.backward()
        optimizer.step()
        return metrics


def train_and_test(dataset, label_rate, eps, experiment_dir, args):
    # define model
    model = NodeClassifier(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        **vars(args)
    )

    # apply transforms
    dataset = LabelRate(rate=label_rate)(dataset)
    dataset = FeatureTransform(method=args.method, eps=eps)(dataset)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        device=args.device,
        checkpoint_dir=os.path.join('checkpoints', experiment_dir),
        logger=TensorBoardLogger(save_dir=os.path.join('logs', experiment_dir)) if args.log else None
    )
    model = trainer.fit(model, dataset)
    result = trainer.test(model, dataset)

    return result['test_acc']


def batch_train_and_test(args):
    dataset = load_dataset(name=args.dataset, feature_range=(0, 1), sparse=True).to(args.device)
    configs = list(product(args.label_rates, args.epsilons))

    for lr, eps in configs:
        experiment_dir = os.path.join(
            f'task:train', f'dataset:{args.dataset}', f'labelrate:{lr}', f'method:{args.method}',
            f'eps:{eps}', f'step:{args.step}', f'agg:{args.aggregator}', f'selfloops:{args.self_loops}'
        )

        results = []
        run_desc = colored_text(experiment_dir.replace('/', ', '), color='green')
        progbar = tqdm(range(args.repeats), desc=run_desc, file=sys.stdout)
        for run in progbar:
            result = train_and_test(
                dataset=dataset, label_rate=lr, eps=eps, args=args,
                experiment_dir=os.path.join(experiment_dir, str(run))
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
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=available_datasets(), required=True)
    parser.add_argument('-m', '--method', type=str, choices=FeatureTransform.available_methods(), required=True)
    parser.add_argument('-e', '--epsilons', nargs='*', type=float, default=[0.0])
    parser.add_argument('-l', '--label-rates', nargs='*', type=float, default=[1.0])
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./output')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser = NodeClassifier.add_module_specific_args(parser)
    args = parser.parse_args()

    if args.method in FeatureTransform.private_methods and min(args.epsilons) <= 0:
        parser.error('LDP method requires eps > 0.')

    if not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.device = 'cpu'

    print_args(args)
    start = time.time()
    batch_train_and_test(args)
    end = time.time()
    print('\nTotal time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
