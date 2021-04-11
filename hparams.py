from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product
from utils import print_args, JobManager


def generate_commands(args, params):
    default_args = f' -s {args.seed} -r {args.repeats} --log --log-mode collective --project-name {args.project} '
    cmds = []
    configs = product(*[[f'{param_name} {param_value}' for param_value in param_range]
                        for param_name, param_range in params.items()])

    for conf in configs:
        command = f'python main.py {default_args}'
        command += ' '.join(conf)
        cmds.append(command)

    return cmds


def create_jobs(args):
    run_cmds = []

    #non-private and fully-private methods
    run_cmds += generate_commands(args, {
        '--dataset': ['cora', 'pubmed', 'facebook', 'lastfm'],
        '--feature': ['raw', 'rnd', 'one', 'ohd'],
        '--learning-rate': [0.01, 0.001, 0.0001],
        '--weight-decay': [0, 1e-4, 1e-3, 1e-2],
        '--dropout': [0, 0.25, 0.5, 0.75]
    })

    # LPGNN
    x_steps = {'cora': 16, 'pubmed': 16, 'facebook': 4, 'lastfm': 8}
    y_steps = {'cora': 8, 'pubmed': 4, 'facebook': 4, 'lastfm': 8}

    for dataset in ['cora', 'pubmed', 'facebook', 'lastfm']:
        for x_eps, y_eps in [
            (1,0),
            (0,1),
            (1,1)
        ]:
            params = {
                '--dataset': [dataset],
                '--learning-rate': [0.01, 0.001, 0.0001],
                '--weight-decay': [0, 1e-4, 1e-3, 1e-2],
                '--dropout': [0, 0.25, 0.5, 0.75],
            }
            if x_eps:
                params.update({
                    '--x-eps': [x_eps],
                    '--x-steps': [x_steps[dataset]]
                })
            if y_eps:
                params.update({
                    '--y-eps': [y_eps],
                    '--y-steps': [y_steps[dataset]],
                    '--propagate-predictions': ['true']
                })

            run_cmds += generate_commands(args, params)

    return run_cmds


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser, parser_create = JobManager.register_arguments(parser)
    parser_create.add_argument('--project', type=str, default='LPGNN-hparams',
                               help='project name for wandb logging')
    parser_create.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
    parser_create.add_argument('-r', '--repeats', type=int, default=10,
                               help="number of times the experiment is repeated")
    args = parser.parse_args()
    print_args(args)

    JobManager(args, cmd_generator=create_jobs).run()


if __name__ == '__main__':
    main()