from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product

import numpy as np
import pandas as pd
from utils import print_args, JobManager


class HyperParams:
    def __init__(self, path):
        self.df = pd.read_pickle(path)

    def get(self, dataset, feature='raw', x_eps=np.inf, y_eps=np.inf):
        if feature == 'crnd':
            feature = 'rnd'
        if x_eps < np.inf:
            x_eps = 1
        if y_eps < np.inf:
            y_eps = 1

        return self.df.loc[
            feature, x_eps, y_eps, dataset
        ][['learning_rate', 'weight_decay', 'dropout']].to_dict()


def generate_command(args, options):
    default_args = f' -s {args.seed} -r {args.repeats} -o results --log --log-mode collective --project-name {args.project} '
    command = f'python main.py {default_args} {options}'
    return command


def get_option_string(dataset, feature, mechanism, model, x_eps, x_steps, y_eps, y_steps, learning_rate, weight_decay, dropout):
    return f"--dataset {dataset} --feature {feature} --mechanism {mechanism} --model {model} " \
           f"--x-eps {x_eps} --x-steps {x_steps} --y-eps {y_eps} --y-steps {y_steps} " \
           f"--learning-rate {learning_rate} --weight-decay {weight_decay} --dropout {dropout}"


def experiment_commands(args):
    run_cmds = []
    hparams = HyperParams(path='hparams/hparams.pkl')

    ## LPGNN ALL CASES

    datasets = ['cora', 'pubmed', 'facebook', 'lastfm']
    x_eps_list = [0.01, 0.1, 1, 2, 3, np.inf]
    x_steps_list = [0, 2, 4, 8, 16]
    y_eps_list = [1, 2, 3, 4, np.inf]
    y_steps_list = [0, 2, 4, 8, 16]

    for dataset, x_eps, x_steps, y_eps, y_steps in product(datasets, x_eps_list, x_steps_list, y_eps_list, y_steps_list):
        params = hparams.get(dataset=dataset, feature='raw', x_eps=x_eps, y_eps=y_eps)
        options = get_option_string(dataset=dataset, feature='raw', mechanism='mbm', model='sage',
                                    x_eps=x_eps, x_steps=x_steps, y_eps=y_eps, y_steps=y_steps, **params)
        command = generate_command(args, options)
        run_cmds.append(command)


    ## FULLY-PRIVATE BASELINES

    datasets = ['cora', 'pubmed', 'facebook', 'lastfm']
    features = ['rnd', 'crnd', 'one', 'ohd']

    for dataset, feature in product(datasets, features):
        params = hparams.get(dataset=dataset, feature=feature)
        options = get_option_string(dataset=dataset, feature=feature, mechanism='mbm', model='sage',
                                    x_eps=np.inf, x_steps=0, y_eps=np.inf, y_steps=0, **params)
        command = generate_command(args, options)
        run_cmds.append(command)


    run_cmds = list(set(run_cmds))
    return run_cmds


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser, parser_create = JobManager.register_arguments(parser)
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