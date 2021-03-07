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
from utils import colored_text, print_args, seed_everything, WandbLogger, add_parameters_as_argument, measure_runtime


@measure_runtime
def run(args):
    dataset = load_dataset(**vars(args))

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
            # define model
            model = NodeClassifier(
                input_dim=dataset.num_features,
                num_classes=dataset.num_classes,
                **vars(args)
            )

            # perturb features
            dataset = Privatize(method=args.method, epsilon=args.epsilon, input_range=args.data_range)(dataset)

            # train the model
            trainer = Trainer(**vars(args), logger=logger)
            trainer.fit(model, dataset)
            result = trainer.test(dataset)

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
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed:
        seed_everything(args.seed)

    run(args)


if __name__ == '__main__':
    main()
