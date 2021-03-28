import os
import sys
import traceback
import uuid
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import copy
import logging
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch_geometric.transforms import Compose
from datasets import load_dataset
from models import NodeClassifier
from trainer import Trainer
from transforms import FeatureTransform, FeaturePerturbation, LabelPerturbation
from utils import print_args, seed_everything, WandbLogger, \
    add_parameters_as_argument, measure_runtime, from_args, str2bool, Enum, EnumAction


class LogMode(Enum):
    INDIVIDUAL = 'individual'
    COLLECTIVE = 'collective'


@measure_runtime
def run(args):
    dataset = from_args(load_dataset, args)

    test_results = []
    val_results = []
    run_id = str(uuid.uuid1())

    logger = None
    if args.log_mode == LogMode.COLLECTIVE:
        logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, reinit=False, group=run_id)

    progbar = tqdm(range(args.repeats), file=sys.stdout)
    for version in progbar:

        if args.log_mode == LogMode.INDIVIDUAL:
            args.version = version
            logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, group=run_id)

        try:
            data = copy(dataset).to(args.device)
            # define model
            model = from_args(NodeClassifier, args, input_dim=data.num_features, num_classes=data.num_classes)

            # preprocess data
            data = Compose([
                from_args(FeatureTransform, args),
                from_args(FeaturePerturbation, args),
                from_args(LabelPerturbation, args)
            ])(data)

            # train the model
            trainer_logger = logger if args.log_mode == LogMode.INDIVIDUAL else None
            trainer = from_args(Trainer, args, logger=trainer_logger)
            best_val_loss = trainer.fit(model, data)
            result = trainer.test(data)

            # process results
            val_results.append(best_val_loss)
            test_results.append(result['test_acc'])
            progbar.set_postfix({'last_test_acc': test_results[-1], 'avg_test_acc': np.mean(test_results)})

        except Exception as e:
            error = ''.join(traceback.format_exception(Exception, e, e.__traceback__))
            logger.log({'error': error})
            raise e
        finally:
            if args.log_mode == LogMode.INDIVIDUAL:
                logger.finish()

    # save results
    if args.log_mode == LogMode.COLLECTIVE:
        logger.log_summary({'best_val_loss': np.mean(val_results), 'test_acc': np.mean(test_results)})

    os.makedirs(args.output_dir, exist_ok=True)
    df_results = pd.DataFrame(test_results, columns=['test_acc']).rename_axis('version').reset_index()
    for arg_name, arg_val in vars(args).items():
        df_results[arg_name] = [arg_val] * len(test_results)
    df_results.to_csv(os.path.join(args.output_dir, f'{run_id}.csv'), index=False)


def main():
    logging.basicConfig(level=logging.INFO)
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    add_parameters_as_argument(load_dataset, group_dataset)

    # data transformation args
    group_perturb = init_parser.add_argument_group(f'data transformation arguments')
    add_parameters_as_argument(FeatureTransform, group_perturb)
    add_parameters_as_argument(FeaturePerturbation, group_perturb)
    add_parameters_as_argument(LabelPerturbation, group_perturb)

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
    group_expr.add_argument('--log', type=str2bool, nargs='?', const=True, default=False, help='enable logging')
    group_expr.add_argument('--log-mode', type=LogMode, action=EnumAction, default=LogMode.INDIVIDUAL,
                            help='logging mode')
    group_expr.add_argument('--project-name', type=str, default='LPGNN', help='project name for wandb logging')

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed:
        seed_everything(args.seed)

    run(args)


if __name__ == '__main__':
    main()
