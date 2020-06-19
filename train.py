import logging

logging.captureWarnings(True)

from colorama import Fore, Style
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

from datasets import load_dataset, GraphLoader, get_available_datasets
from privacy import privatize, get_available_mechanisms
from models import NodeClassifier, LinkPredictor
from utils import TrainOnlyProgressBar, PandasLogger


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
        params = {'input_dim': data.num_features}
        if self.hparams.task == 'node':
            params['num_classes'] = data.num_classes
        model = self.task_models[self.hparams.task](**params, hparams=self.hparams)
        dataloader = GraphLoader(data)
        self.trainer.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)

    def test(self, data):
        dataloader = GraphLoader(data)
        self.trainer.test(test_dataloaders=dataloader)


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
        print(Fore.BLUE + params_str + Style.RESET_ALL)
        logger.log_params(params)

        data_priv = privatize(data, method=method, eps=eps)
        t = GraphTask(logger, hparams)
        t.train(data_priv)
        t.test(data_priv)


def batch_train_and_test(hparams):
    data = load_dataset(hparams.dataset, split_edges=(hparams.task == 'link'), device=hparams.device)
    for method in hparams.methods:
        experiment_name = f'{hparams.task}_{hparams.dataset}_{method}'
        with PandasLogger(
            output_dir=hparams.output_dir,
            experiment_name=experiment_name,
            write_mode='replace'
        ) as logger:
            for eps in hparams.eps_list:
                train_and_test(
                    task=hparams.task,
                    data=data,
                    method=method,
                    eps=eps,
                    hparams=hparams,
                    repeats=hparams.repeats,
                    logger=logger,
                )


def main():
    seed_everything(12345)
    parser = ArgumentParser()

    parser.add_argument('-t', '--task', type=str, choices=GraphTask.get_available_tasks(), required=True)
    parser.add_argument('-d', '--dataset', type=str, choices=get_available_datasets(), required=True)
    parser.add_argument('-m', '--methods', nargs='+', choices=get_available_mechanisms()+['raw'], required=True)
    parser.add_argument('-e', '--eps', nargs='*', type=float, dest='eps_list')
    parser.add_argument('-r', '--repeats', type=int, default=10)
    parser.add_argument('-o', '--output-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])

    # add args based on the task
    temp_args, _ = parser.parse_known_args()
    parser = GraphTask.add_task_specific_args(task_name=temp_args.task, parent_parser=parser)

    args = parser.parse_args()
    print(args)

    batch_train_and_test(args)


if __name__ == '__main__':
    main()
