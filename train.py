import logging
from argparse import ArgumentParser
import os
import time
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datasets import load_dataset, get_available_datasets
from privacy import privatize, get_available_mechanisms
from models import NodeClassifier, LinkPredictor
from utils import TermColors


available_tasks = {
    'node': NodeClassifier,
    'link': LinkPredictor
}


def train_and_test(task, data, method, eps, hparams, repeats, save_dir):
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

        experiment_name = f'{task}_{data.name}_{method}_{eps}'
        logger = TensorBoardLogger(save_dir=save_dir, name=experiment_name, version=run)

        checkpoint_path = os.path.join('checkpoints', experiment_name)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path)

        params = vars(hparams)
        model = available_tasks[task](**params)

        trainer = Trainer.from_argparse_args(
            args=hparams,
            gpus=int(hparams.device == 'cuda' and torch.cuda.is_available()),
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            log_save_interval=1000,
            weights_summary=None,
            deterministic=True,
            progress_bar_refresh_rate=10,
            early_stop_callback=EarlyStopping(min_delta=hparams.min_delta, patience=hparams.patience),
        )

        data_priv = privatize(data, method=method, eps=eps)
        model.fit(data=data_priv, trainer=trainer)
        model.test(data=data_priv, trainer=trainer, ckpt_path='best')


def batch_train_and_test(args):
    data = load_dataset(args.dataset, split_edges=(args.task == 'link'), device=args.device)
    for method in args.methods:
        for eps in args.eps_list:
            train_and_test(
                task=args.task,
                data=data,
                method=method,
                eps=eps,
                hparams=args,
                repeats=args.repeats,
                save_dir=args.output_dir
            )


def main():
    seed_everything(12345)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

    parser = ArgumentParser()

    parser.add_argument('-t', '--task', type=str, choices=available_tasks, required=True,
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
                        help='The number of repeating the experiment. Default is 10.'
                        )
    parser.add_argument('-o', '--output-dir', type=str, default='./results',
                        help='The path to store the results. Default is "./results".'
                        )
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='The device used for the training. Either "cpu" or "cuda". Default is "cuda".'
                        )

    # add args based on the task
    temp_args, _ = parser.parse_known_args()
    parser = available_tasks[temp_args.task].add_module_specific_args(parser)

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
