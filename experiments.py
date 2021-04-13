import os.path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product

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


def get_experiment_cmd(dataset, feature, mechanism, model, x_eps, x_steps, y_eps, y_steps, forward_correction,
                       learning_rate, weight_decay, dropout, args):
    return f"python main.py --dataset {dataset} --feature {feature} --mechanism {mechanism} --model {model} " \
           f"--x-eps {x_eps} --x-steps {x_steps} --y-eps {y_eps} --y-steps {y_steps} " \
           f"--forward-correction {forward_correction} --learning-rate {learning_rate} " \
           f"--weight-decay {weight_decay} --dropout {dropout} -s {args.seed} -r {args.repeats} " \
           f"-o {args.output_dir} --log --log-mode collective --project-name {args.project}"


def experiment_commands(args):
    run_cmds = []
    hparams = HyperParams(path_dir='./hparams')

    # ## LPGNN ALL CASES
    #
    # datasets = ['cora', 'pubmed', 'facebook', 'lastfm']
    # x_eps_list = [0.01, 0.1, 1, 2, 3, np.inf]
    # x_steps_list = [0, 2, 4, 8, 16]
    # y_eps_list = [1, 2, 3, 4, np.inf]
    # y_steps_list = [0, 2, 4, 8, 16]
    #
    # for dataset, x_eps, x_steps, y_eps, y_steps in product(datasets, x_eps_list, x_steps_list, y_eps_list,
    #                                                        y_steps_list):
    #     params = hparams.get(dataset=dataset, feature='raw', x_eps=x_eps, y_eps=y_eps)
    #     command = get_experiment_cmd(
    #         dataset=dataset, feature='raw', mechanism='mbm', model='sage',
    #         x_eps=x_eps, x_steps=x_steps, y_eps=y_eps, y_steps=y_steps,
    #         forward_correction=True, learning_rate=params['learning_rate'],
    #         weight_decay=params['weight_decay'], dropout=params['dropout'], args=args
    #     )
    #     run_cmds.append(command)

    ## FULLY-PRIVATE BASELINES

    datasets = ['cora', 'pubmed', 'facebook', 'lastfm']
    features = ['rnd', 'crnd', 'one', 'ohd']

    for dataset, feature in product(datasets, features):
        params = hparams.get(dataset=dataset, feature=feature, x_eps=np.inf, y_eps=np.inf)
        command = get_experiment_cmd(
            dataset=dataset, feature=feature, mechanism='mbm', model='sage',
            x_eps=np.inf, x_steps=params['x_steps'], y_eps=np.inf, y_steps=params['y_steps'], forward_correction=True,
            learning_rate=params['learning_rate'], weight_decay=params['weight_decay'],
            dropout=params['dropout'], args=args
        )
        run_cmds.append(command)

    ## BASELINE LDP MECHANISMS

    datasets = ['cora', 'pubmed', 'facebook', 'lastfm']
    mechanisms = ['1bm', 'lpm', 'agm']
    x_eps_list = [0.01, 0.1, 1, 2, 3]

    for dataset, mechanism, x_eps in product(datasets, mechanisms, x_eps_list):
        params = hparams.get(dataset=dataset, feature='raw', x_eps=x_eps, y_eps=np.inf)
        command = get_experiment_cmd(
            dataset=dataset, feature='raw', mechanism=mechanism, model='sage',
            x_eps=x_eps, x_steps=params['x_steps'], y_eps=np.inf, y_steps=params['y_steps'],
            forward_correction=True, learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'], dropout=params['dropout'], args=args
        )
        run_cmds.append(command)

    ## NO LABEL CORRECTION

    datasets = ['cora', 'pubmed', 'facebook', 'lastfm']
    y_eps_list = [1, 2, 3]

    for dataset, y_eps in product(datasets, y_eps_list):
        params = hparams.get(dataset=dataset, feature='raw', x_eps=np.inf, y_eps=y_eps)
        command = get_experiment_cmd(
            dataset=dataset, feature='raw', mechanism='mbm', model='sage',
            x_eps=np.inf, x_steps=params['x_steps'], y_eps=y_eps, y_steps=0,
            forward_correction=False, learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'], dropout=params['dropout'], args=args
        )
        run_cmds.append(command)

    run_cmds = list(set(run_cmds))
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
