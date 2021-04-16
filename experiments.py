import os.path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product

import random
import numpy as np
import pandas as pd
from utils import print_args, JobManager


class HyperParams:
    def __init__(self, path_dir):
        self.df_params = pd.read_csv(os.path.join(path_dir, 'params.csv'),
                                     index_col=['dataset_name', 'feature', 'x_eps', 'y_eps'])
        self.df_steps = pd.read_csv(os.path.join(path_dir, 'steps.csv'),
                                    index_col=['dataset_name', 'x_eps', 'y_eps'])

    def get(self, dataset, feature, x_eps, y_eps):
        if feature == 'crnd': feature = 'rnd'
        set_eps = lambda eps: np.inf if np.isinf(x_eps) else 1
        hparams = self.df_params.loc[dataset, feature, set_eps(x_eps), set_eps(y_eps)].to_dict()
        steps = self.df_steps.loc[dataset, x_eps, y_eps].to_dict()
        hparams.update(steps)

        return hparams


class CommandBuilder:
    BEST_VAL = None

    def __init__(self, args, hparams_dir=None, random=None):
        self.random = random
        self.default_options = f" -s {args.seed} -r {args.repeats} -o {args.output_dir} --log --log-mode collective " \
                               f"--project-name {args.project}"
        self.hparams = HyperParams(path_dir=hparams_dir) if hparams_dir else None

    def build(self, dataset, feature, mechanism, model, x_eps, y_eps, forward_correction,
              x_steps, y_steps, lambdaa, learning_rate, weight_decay, dropout):

        cmd_list = []
        configs = self.product_dict(
            dataset=self.get_list(dataset),
            feature=self.get_list(feature),
            mechanism=self.get_list(mechanism),
            model=self.get_list(model),
            x_eps=self.get_list(x_eps),
            y_eps=self.get_list(y_eps),
            forward_correction=self.get_list(forward_correction),
            x_steps=self.get_list(x_steps),
            y_steps=self.get_list(y_steps),
            lambdaa=self.get_list(lambdaa),
            learning_rate=self.get_list(learning_rate),
            weight_decay=self.get_list(weight_decay),
            dropout=self.get_list(dropout),
        )

        if self.random:
            configs = random.sample(list(configs), self.random)

        for config in configs:
            config = self.fill_best_params(config)
            options = ' '.join([f' --{param} {value} ' for param, value in config.items()])
            command = f'python main.py {options} {self.default_options}'
            cmd_list.append(command)

        return cmd_list

    def fill_best_params(self, config):
        if self.hparams:
            best_params = self.hparams.get(
                dataset=config['dataset'],
                feature=config['feature'],
                x_eps=config['x_eps'],
                y_eps=config['y_eps']
            )

            for param, value in config.items():
                if value == self.BEST_VAL:
                    config[param] = best_params[param]
        return config

    @staticmethod
    def get_list(param):
        if not (isinstance(param, list) or isinstance(param, tuple)):
            param = list(param)
        return param

    @staticmethod
    def product_dict(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))


def experiment_commands(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams', random=100)
    datasets = ['cora', 'pubmed', 'facebook', 'lastfm']

    run_cmds += cmdbuilder.build(
        dataset='cora',
        feature='raw',
        mechanism='mbm',
        model='sage',
        x_eps=1,
        x_steps=[16],
        y_eps=[1],
        y_steps=[0, 2, 4, 8, 16],
        forward_correction=True,
        lambdaa=np.arange(0, 0.51, 0.1),
        learning_rate=[0.01, 0.001, 0.0001],
        weight_decay=[0.01, 0.001, 0.0001],
        dropout=np.arange(0, 1, 0.25)
    )

    # ## LPGNN ALL CASES
    # run_cmds += cmdbuilder.build(
    #     dataset=datasets,
    #     feature='raw',
    #     mechanism='mbm',
    #     model='sage',
    #     x_eps=[0.01, 0.1, 1, 2, 3, np.inf],
    #     x_steps=[0, 2, 4, 8, 16],
    #     y_eps=[1, 2, 3, 4, np.inf],
    #     y_steps=[0, 2, 4, 8, 16],
    #     forward_correction=True,
    #     lambdaa=CommandBuilder.BEST_VAL,
    #     learning_rate=CommandBuilder.BEST_VAL,
    #     weight_decay=CommandBuilder.BEST_VAL,
    #     dropout=CommandBuilder.BEST_VAL
    # )
    #
    # ## FULLY-PRIVATE BASELINES
    # run_cmds += cmdbuilder.build(
    #     dataset=datasets,
    #     feature=['rnd', 'crnd', 'one', 'ohd'],
    #     mechanism='mbm',
    #     model='sage',
    #     x_eps=np.inf,
    #     x_steps=CommandBuilder.BEST_VAL,
    #     y_eps=np.inf,
    #     y_steps=CommandBuilder.BEST_VAL,
    #     forward_correction=True,
    #     lambdaa=CommandBuilder.BEST_VAL,
    #     learning_rate=CommandBuilder.BEST_VAL,
    #     weight_decay=CommandBuilder.BEST_VAL,
    #     dropout=CommandBuilder.BEST_VAL
    # )
    #
    # ## BASELINE LDP MECHANISMS
    # run_cmds += cmdbuilder.build(
    #     dataset=datasets,
    #     feature='raw',
    #     mechanism=['1bm', 'lpm', 'agm'],
    #     model='sage',
    #     x_eps=[0.01, 0.1, 1, 2, 3],
    #     x_steps=CommandBuilder.BEST_VAL,
    #     y_eps=np.inf,
    #     y_steps=CommandBuilder.BEST_VAL,
    #     forward_correction=True,
    #     lambdaa=CommandBuilder.BEST_VAL,
    #     learning_rate=CommandBuilder.BEST_VAL,
    #     weight_decay=CommandBuilder.BEST_VAL,
    #     dropout=CommandBuilder.BEST_VAL
    # )
    #
    # ## NO LABEL CORRECTION
    # run_cmds += cmdbuilder.build(
    #     dataset=datasets,
    #     feature='raw',
    #     mechanism='mbm',
    #     model='sage',
    #     x_eps=np.inf,
    #     x_steps=CommandBuilder.BEST_VAL,
    #     y_eps=[1, 2, 3],
    #     y_steps=CommandBuilder.BEST_VAL,
    #     forward_correction=False,
    #     lambdaa=CommandBuilder.BEST_VAL,
    #     learning_rate=CommandBuilder.BEST_VAL,
    #     weight_decay=CommandBuilder.BEST_VAL,
    #     dropout=CommandBuilder.BEST_VAL
    # )

    run_cmds = list(set(run_cmds))  # remove duplicate runs
    return run_cmds


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser, parser_create = JobManager.register_arguments(parser)
    parser.add_argument('-o', '--output-dir', type=str, default='./results', help="directory to store the results")
    parser_create.add_argument('--project', type=str, default='LPGNN-experiments',
                               help='project name for wandb logging')
    parser_create.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
    parser_create.add_argument('-r', '--repeats', type=int, default=10,
                               help="number of times the experiment is repeated")
    args = parser.parse_args()
    print_args(args)

    JobManager(args, cmd_generator=experiment_commands).run()


if __name__ == '__main__':
    main()
