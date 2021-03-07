import os
import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from models import NodeClassifier
from trainer import Trainer
from transforms import Privatize
from utils import colored_text, print_args, seed_everything, TensorBoardLogger, add_parameters_as_argument, measure_runtime


@measure_runtime
def run(args):

    dataset = load_dataset(**vars(args)).to(args.device)

    experiment_dir = os.path.join(
        f'task:train', f'dataset:{args.dataset_name}', f'labelrate:{args.train_ratio}', f'method:{args.method}',
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
        dataset = Privatize(method=args.method, epsilon=args.epsilon, input_range=args.data_range)(dataset)

        trainer = Trainer(
            **vars(args),
            logger=TensorBoardLogger(save_dir=os.path.join('logs', experiment_dir, str(run_id))) if args.log else None
        )

        trainer.fit(model, dataset)
        result = trainer.test(dataset)
        results.append(result['test_acc'])
        progbar.set_postfix({'last_test_acc': results[-1], 'avg_test_acc': np.mean(results)})

    # save results
    save_dir = os.path.join(args.output_dir, experiment_dir)
    os.makedirs(save_dir, exist_ok=True)
    df_results = pd.DataFrame(results, columns=['test_acc']).rename_axis('version').reset_index()
    df_results.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)


def main():
    seed_everything(12345)
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    add_parameters_as_argument(load_dataset, group_dataset)

    # perturbation arguments
    group_perturb = init_parser.add_argument_group(f'perturbation arguments')
    add_parameters_as_argument(Privatize.__init__, group_perturb)

    # model args
    group_model = init_parser.add_argument_group(f'model arguments')
    add_parameters_as_argument(NodeClassifier.__init__, group_model)

    # trainer arguments (depends on perturbation)
    group_trainer = init_parser.add_argument_group(f'trainer arguments')
    add_parameters_as_argument(Trainer.__init__, group_trainer)

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=None, help='initial random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=1, help="number of times the experiment is repeated")
    group_expr.add_argument('-o', '--output-dir', type=str, default='./output', help="directory to store the results")
    group_expr.add_argument('--log', action='store_true', help='enable logging')

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)

    if args.seed:
        seed_everything(args.seed)

    run(args)


if __name__ == '__main__':
    main()
