import os
from argparse import ArgumentParser
from itertools import product
from datasets import supported_datasets
from transforms import Privatize
from utils import print_args

parser = ArgumentParser()
parser.add_argument('--project-name', type=str, default='LPGNN-Tune', help='project name for wandb logging')
parser.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
parser.add_argument('-r', '--repeats', type=int, default=10, help="number of times the experiment is repeated")
subparser = parser.add_subparsers()
parser_grid = subparser.add_parser('grid')
parser_grid.add_argument('-q', '--queue', type=str, default='sgpu', choices=['sgpu', 'gpu', 'lgpu'])
parser_grid.add_argument('-m', '--gpumem', type=int, default=10)
parser_grid.add_argument('-j', '--jobs-dir', type=str, default='./jobs')
args = parser.parse_args()
print_args(args)

default_args = f' -s {args.seed} -r {args.repeats} --log --log-mode collective --project-name {args.project_name} '

params = {
    '--dataset': list(supported_datasets),
    '--method': list(Privatize.non_private_methods),
    '--learning-rate': [0.001, 0.01, 0.1],
    '--weight-decay': [0, 1e-4, 1e-3, 1e-2],
    '--dropout': [0, 0.25, 0.5]
}

run_cmds = []
configs = product(*[[f'{param_name} {param_value}' for param_value in param_range]
                    for param_name, param_range in params.items()])

for conf in configs:
    command = f'python main.py {default_args}'
    command += ' '.join(conf)
    run_cmds.append(command)

if 'queue' in args:
    os.makedirs(args.jobs_dir, exist_ok=True)
    for i, run in enumerate(run_cmds):
        job_file_content = [
            f'#$ -N job-{i + 1}\n',
            f'#$ -S /bin/bash\n',
            f'#$ -P socialcomputing\n',
            f'#$ -l pytorch,{args.queue},gpumem={args.gpumem}\n',
            f'#$ -cwd\n',
            f'cd ..\n',
            f'{run}\n'
        ]
        with open(os.path.join(args.jobs_dir, f'job-{i + 1}.job'), 'w') as file:
            file.writelines(job_file_content)
            file.flush()

    print('Job files created in:', args.jobs_dir)
else:
    for run in run_cmds:
        # print(run)
        os.system(run)
