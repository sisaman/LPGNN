import os
import sys
import traceback
import uuid
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch_geometric.transforms import Compose
from datasets import load_dataset
from models import NodeClassifier
from trainer import Trainer
from transforms import FeatureTransform, FeaturePerturbation, LabelPerturbation
from utils import print_args, seed_everything, WandbLogger, \
    add_parameters_as_argument, measure_runtime, from_args, str2bool, Enum, EnumAction, colored_text


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
            data = dataset.clone().to(args.device)

            # preprocess data
            data = Compose([
                from_args(FeatureTransform, args),
                from_args(FeaturePerturbation, args),
                from_args(LabelPerturbation, args)
            ])(data)

            # define model
            model = from_args(NodeClassifier, args, input_dim=data.num_features, num_classes=data.num_classes)

            # train the model
            trainer = from_args(Trainer, args, logger=logger if args.log_mode == LogMode.INDIVIDUAL else None)
            best_metrics = trainer.fit(model, data)
            result = trainer.test(data)

            # process results
            val_results.append(best_metrics['val/acc'])
            test_results.append(result['test/acc'])
            progbar.set_postfix({'last_test_acc': test_results[-1], 'avg_test_acc': np.mean(test_results)})

        except Exception as e:
            error = ''.join(traceback.format_exception(Exception, e, e.__traceback__))
            logger.log_summary({'error': error})
            raise e
        finally:
            if args.log_mode == LogMode.INDIVIDUAL:
                logger.finish()

    if args.log_mode == LogMode.COLLECTIVE:
        logger.log_summary({
            'val/acc_mean': np.mean(val_results),
            'val/acc_std': np.std(val_results),         # todo replace with CI
            'test/acc_mean': np.mean(test_results),
            'test/acc_std': np.std(test_results)        # todo replace with CI
        })

    if not args.log:
        os.makedirs(args.output_dir, exist_ok=True)
        df_results = pd.DataFrame(test_results, columns=['test_acc']).rename_axis('version').reset_index()
        df_results['group'] = run_id
        for arg_name, arg_val in vars(args).items():
            df_results[arg_name] = [arg_val] * len(test_results)
        df_results.to_csv(os.path.join(args.output_dir, f'{run_id}.csv'), index=False)


def main():
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
    group_trainer.add_argument('--device', help='desired device for training', choices=['cpu', 'cuda'], default='cuda')

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

    if args.device == 'cuda' and not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.device = 'cpu'

    try:
        run(args)
    except KeyboardInterrupt:
        print('Graceful Shutdown...')


if __name__ == '__main__':
    main()
