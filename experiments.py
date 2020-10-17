import os
from argparse import ArgumentParser
from itertools import product

from utils import print_args, colored_text

parser = ArgumentParser()
parser.add_argument('-r', '--repeats', type=int, default=10)
parser.add_argument('-o', '--output-dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
subparser = parser.add_subparsers()
parser_grid = subparser.add_parser('grid')
parser_grid.add_argument('-q', '--queue', type=str, default='sgpu', choices=['sgpu', 'gpu', 'lgpu'])
parser_grid.add_argument('-m', '--gpumem', type=int, default=8)
parser_grid.add_argument('-j', '--jobs-dir', type=str, default='./jobs')
args = parser.parse_args()
print_args(args)

datasets = {
    'cora':     {'--learning-rate': 0.01, '--weight-decay': 0.010, '--dropout': 0.00},
    'citeseer': {'--learning-rate': 0.01, '--weight-decay': 0.010, '--dropout': 0.00},
    'pubmed':   {'--learning-rate': 0.01, '--weight-decay': 0.001, '--dropout': 0.00},
    'facebook': {'--learning-rate': 0.01, '--weight-decay': 0.001, '--dropout': 0.50},
    'github':   {'--learning-rate': 0.01, '--weight-decay': 0.000, '--dropout': 0.50},
    'lastfm':   {'--learning-rate': 0.01, '--weight-decay': 0.001, '--dropout': 0.75},
}

# EFFECT OF MULTI-BIT MECHANISM
error_run = f"python error.py -d {' '.join(datasets.keys())} -m agm obm mbm -e 0.1 0.5 1 2 -a mean gcn"
print(colored_text(error_run, color='lightcyan'))
os.system(error_run)

configs = set()

# PRIVACY-UTILITY EXPERIMENTS
eps_range = [0.01, 0.1, 0.5, 1, 2]
step_range = [1, 2, 4, 8, 16, 32]

configs |= {'-l 1 -m raw -e 0 -k 1 -a gcn --self-loops'}  # GCN+Raw
configs |= {'-l 1 -m rnd -e 0 -k 1 -a gcn --self-loops'}  # GCN+Rnd

for eps, k in product(eps_range, step_range):
    configs |= {f'-l 1 -m mbm -e {eps} -k {k} -a gcn --no-self-loops'}  # LPGNN

# EFFECT OF KPROP
eps_range = [0.01, 0.1, 1]
step_range = [1, 2, 4, 8, 16, 32]

for k in step_range:
    configs |= {f'-l 1 -m raw -e 0 -k {k} -a gcn --no-self-loops'}  # Raw

for eps, k in product(eps_range, step_range, ):
    configs |= {f'-l 1 -m mbm -e {eps} -k {k} -a gcn --no-self-loops'}  # without self loops
    configs |= {f'-l 1 -m mbm -e {eps} -k {k} -a gcn --self-loops'}     # with self loops

# EFFECT OF LABEL RATE
lr_range = [0.1, 0.2, 0.3, 0.4, 0.5]
eps_range = [0.01, 0.1, 1]
step_range = [2, 4, 8]

for lr in lr_range:
    for eps in eps_range:
        configs |= {f'-l {lr} -m mbm -e {eps} -k 4 -a gcn --no-self-loops'}
    for k in step_range:
        configs |= {f'-l {lr} -m mbm -e 1 -k {k} -a gcn --no-self-loops'}


train_runs = []
for dataset, hparams in datasets.items():
    command = f'python train.py -d {dataset} '
    command += ' '.join([f'{key} {val}' for key, val in hparams.items()])
    command += f' -r {args.repeats} -o "{args.output_dir}" --device {args.device} '
    for config in configs:
        train_runs.append(command + config)


if 'queue' in args:
    os.makedirs(args.jobs_dir, exist_ok=True)
    for i, run in enumerate(train_runs):
        job_file_content = [
            f'#$ -N job-{i}\n',
            f'#$ -S /bin/bash\n',
            f'#$ -P socialcomputing\n',
            f'#$ -l buster,pytorch,{args.queue},gpumem={args.gpumem}\n',
            f'#$ -cwd\n',
            f'## Task\n',
            f'cd ..\n',
            f'{run}\n'
        ]
        with open(os.path.join(args.jobs_dir, f'job-{i}.job'), 'w') as file:
            file.writelines(job_file_content)
            file.flush()

    print('Job files created in:', args.jobs_dir)
else:
    for run in train_runs:
        print(colored_text(run, color='lightcyan'))
        os.system(run)
