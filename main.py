import os
import sys
import traceback
import uuid
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets import load_dataset
from models import NodeClassifier
from trainer import Trainer
from transforms import Privatize
from utils import colored_text, print_args, seed_everything, WandbLogger, \
    add_parameters_as_argument, measure_runtime, from_args


@measure_runtime
def run(args):

    experiment_name = ', '.join([
        args.dataset_name, args.method, f'label:{args.train_ratio}',
        f'e:{args.epsilon}', f'k:{args.step}', f'agg:{args.aggregator}', f'loop:{int(args.self_loops)}'
    ])

    results = []
    run_desc = colored_text(experiment_name.replace('/', ', '), color='green')
    progbar = tqdm(range(args.repeats), desc=run_desc, file=sys.stdout)
    for run_id in progbar:
        args.version = run_id
        logger = WandbLogger(project='LPGNN', name=experiment_name, config=args, enabled=args.log)

        try:
            data = from_args(load_dataset, args)
            # define model
            model = from_args(NodeClassifier, args, input_dim=data.num_features, num_classes=data.num_classes)

            # perturb features
            data = Privatize(method=args.method, epsilon=args.epsilon, input_range=args.data_range)(data)

            # train the model
            trainer = from_args(Trainer, args, logger=logger)
            trainer.fit(model, data)
            result = trainer.test(data)

            # process results
            results.append(result['test_acc'])
            progbar.set_postfix({'last_test_acc': results[-1], 'avg_test_acc': np.mean(results)})

        except Exception as e:
            error = ''.join(traceback.format_exception(Exception, e, e.__traceback__))
            logger.log({'error': error})
            raise e
        finally:
            logger.finish()

    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    df_results = pd.DataFrame(results, columns=['test_acc']).rename_axis('version').reset_index()
    for arg_name, arg_val in vars(args).items():
        df_results[arg_name] = [arg_val] * len(results)
    df_results.to_csv(os.path.join(args.output_dir, f'{uuid.uuid1()}.csv'), index=False)


def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    add_parameters_as_argument(load_dataset, group_dataset)

    # perturbation arguments
    group_perturb = init_parser.add_argument_group(f'perturbation arguments')
    add_parameters_as_argument(Privatize, group_perturb)

    # model args
    group_model = init_parser.add_argument_group(f'model arguments')
    add_parameters_as_argument(NodeClassifier, group_model)

    # trainer arguments (depends on perturbation)
    group_trainer = init_parser.add_argument_group(f'trainer arguments')
    add_parameters_as_argument(Trainer, group_trainer)

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=None, help='initial random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=1, help="number of times the experiment is repeated")
    group_expr.add_argument('-o', '--output-dir', type=str, default='./output', help="directory to store the results")
    group_expr.add_argument('--log', action='store_true', help='enable logging')

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed:
        seed_everything(args.seed)

    run(args)


if __name__ == '__main__':
    main()
