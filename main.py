import os
import sys
import time
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import supported_datasets, load_dataset
from models import NodeClassifier
from transforms import Privatize
from utils import colored_text, print_args, seed_everything, TensorBoardLogger


def run(args):

    dataset = load_dataset(name=args.dataset, feature_range=(0, 1),
                           sparse=True, train_ratio=args.label_rate).to(args.device)

    experiment_dir = os.path.join(
        f'task:train', f'dataset:{args.dataset}', f'labelrate:{args.label_rate}', f'method:{args.method}',
        f'eps:{args.epsilon}', f'step:{args.step}', f'agg:{args.aggregator}', f'selfloops:{args.self_loops}'
    )

    results = []
    run_desc = colored_text(experiment_dir.replace('/', ', '), color='green')
    progbar = tqdm(range(args.repeats), desc=run_desc, file=sys.stdout)
    for run_id in progbar:
        # define model
        model = NodeClassifier(
            input_dim=dataset.num_features,
            num_classes=dataset.num_classes,
            **vars(args)
        )

        # apply transforms
        dataset = Privatize(method=args.method, eps=args.epsilon)(dataset)

        trainer = Trainer(
            max_epochs=args.max_epochs,
            device=args.device,
            checkpoint_dir=os.path.join('checkpoints', experiment_dir, str(run_id)),
            logger=TensorBoardLogger(save_dir=os.path.join('logs', experiment_dir, str(run_id))) if args.log else None
        )

        model = trainer.fit(model, dataset)
        result = trainer.test(model, dataset)
        results.append(result['test_acc'])
        progbar.set_postfix({'last_test_acc': results[-1], 'avg_test_acc': np.mean(results)})

    # save results
    save_dir = os.path.join(args.output_dir, experiment_dir)
    os.makedirs(save_dir, exist_ok=True)
    df_results = pd.DataFrame(results, columns=['test_acc']).rename_axis('version').reset_index()
    df_results.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)


def main():
    seed_everything(12345)
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=supported_datasets, required=True)
    parser.add_argument('-m', '--method', type=str, choices=Privatize.supported_methods(), required=True)
    parser.add_argument('-e', '--epsilon', type=float, default=0.0)
    parser.add_argument('-l', '--label-rate', type=float, default=0.5)
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-o', '--output-dir', type=str, default='./output')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser = NodeClassifier.add_module_specific_args(parser)
    args = parser.parse_args()

    if args.method in Privatize.private_methods and args.epsilon <= 0:
        parser.error('LDP method requires eps > 0.')

    if not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.device = 'cpu'

    print_args(args)
    start = time.time()
    run(args)
    end = time.time()
    print('\nTotal time spent:', end - start, 'seconds.\n\n')


if __name__ == '__main__':
    main()
