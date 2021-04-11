import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product
from utils import print_args
from tqdm.auto import tqdm
from subprocess import DEVNULL, STDOUT, check_call


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-j', '--jobs-dir', type=str, default='./jobs')
subparser = parser.add_subparsers(dest='command')

parser_create = subparser.add_parser('create')
parser_create.add_argument('--project-name', type=str, default='LPGNN-hparams', help='project name for wandb logging')
parser_create.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
parser_create.add_argument('-r', '--repeats', type=int, default=10, help="number of times the experiment is repeated")

subparser_create = parser_create.add_subparsers()
parser_grid = subparser_create.add_parser('grid')
parser_grid.add_argument('-q', '--queue', type=str, default='sgpu', choices=['sgpu', 'gpu', 'lgpu'])
parser_grid.add_argument('-m', '--gpumem', type=int, default=10)

subparser.add_parser('submit')
subparser.add_parser('status')

parser_resubmit = subparser.add_parser('resubmit')
parser_resubmit.add_argument('--loop', action='store_true')

args = parser.parse_args()
print_args(args)

def generate_commands(params):
    default_args = f' -s {args.seed} -r {args.repeats} --log --log-mode collective --project-name {args.project_name} '
    cmds = []
    configs = product(*[[f'{param_name} {param_value}' for param_value in param_range]
                        for param_name, param_range in params.items()])

    for conf in configs:
        command = f'python main.py {default_args}'
        command += ' '.join(conf)
        cmds.append(command)

    return cmds


def get_failed_jobs():
    file_list = os.listdir()
    failed_jobs = []
    for file in file_list:
        if file.count('.e'):
            num_lines = sum(1 for _ in open(file))
            if num_lines > 0:
                job_file = file.split('.')[0] + '.job'
                failed_jobs.append((job_file, file, num_lines))

    return failed_jobs


def create_jobs():
    run_cmds = []

    # non-private and fully-private methods
    # run_cmds += generate_commands({
    #     '--dataset': ['cora', 'pubmed', 'facebook', 'lastfm'],
    #     '--feature': ['raw', 'rnd', 'one', 'ohd'],
    #     '--learning-rate': [0.01, 0.001, 0.0001],
    #     '--weight-decay': [0, 1e-4, 1e-3, 1e-2],
    #     '--dropout': [0, 0.25, 0.5, 0.75]
    # })

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

            run_cmds += generate_commands(params)



    os.makedirs(args.jobs_dir, exist_ok=True)

    if 'queue' in args:
        for i, run in tqdm(enumerate(run_cmds), total=len(run_cmds)):
            job_file_content = [
                f'#$ -N job-{i + 1}\n',
                f'#$ -S /bin/bash\n',
                f'#$ -P dusk2dawn\n',
                f'#$ -l pytorch,{args.queue},gpumem={args.gpumem}\n',
                f'#$ -cwd\n',
                f'cd ..\n',
                f'{run}\n'
            ]
            with open(os.path.join(args.jobs_dir, f'job-{i + 1}.job'), 'w') as file:
                file.writelines(job_file_content)
                file.flush()
    else:
        with open(os.path.join(args.jobs_dir, f'all.jobs'), 'w') as file:
            for run in tqdm(run_cmds):
                file.write(run + '\n')

    print('Job files created in:', args.jobs_dir)

if args.command == 'create':
    create_jobs()
elif args.command == 'submit':
    os.chdir(args.jobs_dir)
    job_list = [job for job in os.listdir() if job.endswith('.job')]
    for job in tqdm(job_list, desc='submitting jobs'):
        check_call(['qsub', '-V', job], stdout=DEVNULL, stderr=STDOUT)
    print('Done.')
elif args.command == 'status':
    os.chdir(args.jobs_dir)
    failed_jobs = get_failed_jobs()
    for _, file, num_lines in failed_jobs:
        print(num_lines, os.path.join(args.jobs_dir, file))
elif args.command == 'resubmit':
    os.chdir(args.jobs_dir)

    while True:
        try:
            failed_jobs = get_failed_jobs()
            for job_file, error_file, _ in failed_jobs:
                print('resubmitting', job_file, '...', end='')
                check_call(['qsub', '-V', job_file], stdout=DEVNULL, stderr=STDOUT)
                os.remove(error_file)
                print('done')

            if args.loop:
                time.sleep(60)
            else:
                break
        except KeyboardInterrupt:
            break

