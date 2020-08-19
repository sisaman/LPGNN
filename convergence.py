import logging

from pytorch_lightning.loggers import TensorBoardLogger

logging.captureWarnings(True)

from argparse import ArgumentParser

import time
import torch
from pytorch_lightning import Trainer, seed_everything
from datasets import load_dataset, get_available_datasets
from privacy import privatize, get_available_mechanisms
from models import NodeClassifier, LinkPredictor
from utils import TermColors


available_tasks = {
    'node': NodeClassifier,
    'link': LinkPredictor
}


def convergence_test(args):
    data = load_dataset(args.dataset, split_edges=(args.task == 'link'), device=args.device)
    for method in args.methods:
        for eps in args.eps_list:
            params = {
                'task': args.task,
                'dataset': data.name,
                'method': method,
                'eps': eps,
            }
            experiment_name = f'{args.task}_{args.dataset}_{eps}'
            logger = TensorBoardLogger(save_dir="tb_logs", name=experiment_name)

            params_str = ' | '.join([f'{key}={val}' for key, val in params.items()])
            print(TermColors.FG.green + params_str + TermColors.reset)

            data_priv = privatize(data, method=method, eps=eps)

            params = vars(args)
            model = available_tasks[args.task](**params)

            trainer = Trainer.from_argparse_args(
                args=args,
                gpus=int(args.device == 'cuda' and torch.cuda.is_available()),
                checkpoint_callback=False,
                logger=logger,
                weights_summary=None,
                deterministic=True,
                early_stop_callback=False,
            )

            model.fit(data=data_priv, trainer=trainer)


def main():
    seed_everything(12345)
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
                parser.error('LDP require eps > 0.')

    args = parser.parse_args()
    args.check_val_every_n_epoch = 1
    print(args)

    start = time.time()
    convergence_test(args)
    end = time.time()
    print('Total time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
