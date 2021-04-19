import os
import time
from argparse import ArgumentTypeError, Action
import inspect
import enum
import functools
from subprocess import check_call, DEVNULL, STDOUT
from torch_geometric.utils import accuracy as accuracy_1d

import torch
import numpy as np
import random
import torch.nn.functional as F
import seaborn as sns
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None


def accuracy(pred, target):
    pred = pred.argmax(dim=1) if len(pred.size()) > 1 else pred
    target = target.argmax(dim=1) if len(target.size()) > 1 else target
    return accuracy_1d(pred=pred, target=target)


def cross_entropy_loss(p_y, y, weighted=False):
    y_onehot = F.one_hot(y.argmax(dim=1))
    loss = -torch.log(p_y + 1e-20) * y_onehot
    loss *= y if weighted else 1
    loss = loss.sum(dim=1).mean()
    return loss


def js_div(p, q):
    eps = 1e-20
    m = (p + q) / 2
    js = F.kl_div(torch.log(p + eps), m) + F.kl_div(torch.log(q + eps), m)
    return js / 2


def confidence_interval(data, func=np.mean, size=1000, ci=95, seed=12345):
    bs_replicates = sns.algorithms.bootstrap(data, func=func, n_boot=size, seed=seed)
    bounds = sns.utils.ci(bs_replicates, ci)
    return (bounds[1] - bounds[0]) / 2


def measure_runtime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        print(f'\nTotal time spent in {str(func.__name__)}:', end - start, 'seconds.\n\n')
        return out

    return wrapper


class WandbLogger:
    def __init__(self, project=None, name=None, config=None, save_code=True,
                 reinit=True, enabled=True, **kwargs):
        self.enabled = enabled
        if enabled:
            if wandb is None:
                raise ImportError('wandb is not installed yet, install it with `pip install wandb`.')

            os.environ["WANDB_SILENT"] = "true"

            settings = wandb.Settings(start_method="fork")  # noqa

            self.experiment = wandb.init(
                name=name, project=project,
                reinit=reinit, resume='allow', config=config, save_code=save_code,
                settings=settings,
                **kwargs)

    def log(self, metrics):
        if self.enabled:
            self.experiment.log(metrics)

    def log_summary(self, metrics):
        if self.enabled:
            for metric, value in metrics.items():
                self.experiment.summary[metric] = value

    def watch(self, model):
        if self.enabled:
            self.experiment.watch(model, log_freq=50)

    def finish(self):
        if self.enabled:
            self.experiment.finish()


class JobManager:
    def __init__(self, args, cmd_generator=None):
        self.args = args
        self.cmd_generator = cmd_generator

    def run(self):
        if self.args.command == 'create':
            self.create()
        elif self.args.command == 'submit':
            self.submit()
        elif self.args.command == 'status':
            self.status()
        elif self.args.command == 'resubmit':
            self.resubmit()
        elif self.args.command == 'exec':
            self.exec()

    def create_job_array(self, run_cmds):
        self.create_job_list(run_cmds)

        window = 7500
        for i in range(0, len(run_cmds), window):
            begin = i + 1
            end = min(i + window, len(run_cmds))

            job_file_content = [
                f'#$ -N job-{begin}-{end}\n',
                f'#$ -S /bin/bash\n',
                f'#$ -P dusk2dawn\n',
                f'#$ -M sajadmanesh@idiap.ch\n',
                f'#$ -l pytorch,{self.args.queue},gpumem={self.args.gpumem}\n',
                f'#$ -t {begin}-{end}\n',
                f'#$ -cwd\n',
                f'cd ..\n',
                f'python experiments.py exec --id $SGE_TASK_ID \n'
            ]

            with open(os.path.join(self.args.jobs_dir, f'job-{begin}-{end}.job'), 'w') as file:
                file.writelines(job_file_content)
                file.flush()

    def create_individual_jobs(self, run_cmds):
        for i, run in tqdm(enumerate(run_cmds), total=len(run_cmds)):
            job_file_content = [
                f'#$ -N job-{i + 1}\n',
                f'#$ -S /bin/bash\n',
                f'#$ -P dusk2dawn\n',
                f'#$ -l pytorch,{self.args.queue},gpumem={self.args.gpumem}\n',
                f'#$ -cwd\n',
                f'cd ..\n',
                f'{run}\n'
            ]
            with open(os.path.join(self.args.jobs_dir, f'job-{i + 1}.job'), 'w') as file:
                file.writelines(job_file_content)
                file.flush()

    def create_job_list(self, run_cmds):
        with open(os.path.join(self.args.jobs_dir, f'all.jobs'), 'w') as file:
            for run in tqdm(run_cmds):
                file.write(run + '\n')

    def create(self):
        os.makedirs(self.args.jobs_dir, exist_ok=True)
        run_cmds = self.cmd_generator(self.args)

        if 'queue' in self.args:
            if self.args.array:
                self.create_job_array(run_cmds)
            else:
                self.create_individual_jobs(run_cmds)
        else:
            self.create_job_list(run_cmds)

        print('Job files created in:', self.args.jobs_dir)

    def submit(self):
        os.chdir(self.args.jobs_dir)
        job_list = [file for file in os.listdir() if file.endswith('.job')]
        for job in tqdm(job_list, desc='submitting jobs'):
            check_call(['qsub', '-V', job], stdout=DEVNULL, stderr=STDOUT)
        print('Done.')

    def resubmit(self):
        os.chdir(self.args.jobs_dir)

        while True:
            try:
                failed_jobs = self.get_failed_jobs()
                for job_file, error_file, _ in failed_jobs:
                    print('resubmitting', job_file, '...', end='')
                    check_call(['qsub', '-V', job_file], stdout=DEVNULL, stderr=STDOUT)
                    os.remove(error_file)
                    print('done')

                if self.args.loop:
                    time.sleep(60)
                else:
                    break
            except KeyboardInterrupt:
                break

    def status(self):
        os.chdir(self.args.jobs_dir)
        failed_jobs = self.get_failed_jobs()
        for _, file, num_lines in failed_jobs:
            print(num_lines, os.path.join(self.args.jobs_dir, file))

    def exec(self):
        with open(os.path.join(self.args.jobs_dir, 'all.jobs')) as jobs_file:
            job_list = jobs_file.readlines()

        check_call(job_list[self.args.id-1].split())

    @staticmethod
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

    @staticmethod
    def register_arguments(parser, default_jobs_dir='./jobs', default_gpu_mem=10, default_queue='sgpu'):
        parser.add_argument('-j', '--jobs-dir', type=str, default=default_jobs_dir)
        command_subparser = parser.add_subparsers(dest='command')

        parser_create = command_subparser.add_parser('create')
        create_subparser = parser_create.add_subparsers()
        parser_grid = create_subparser.add_parser('grid')
        parser_grid.add_argument('-q', '--queue', type=str, default=default_queue, choices=['sgpu', 'gpu', 'lgpu'])
        parser_grid.add_argument('-m', '--gpumem', type=int, default=default_gpu_mem)
        parser_grid.add_argument('--array', action='store_true')

        command_subparser.add_parser('submit')
        command_subparser.add_parser('status')

        parser_resubmit = command_subparser.add_parser('resubmit')
        parser_resubmit.add_argument('--loop', action='store_true')

        parser_exec = command_subparser.add_parser('exec')
        parser_exec.add_argument('--id', type=int, required=True)
        return parser, parser_create


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def add_parameters_as_argument(function, parser):
    if inspect.isclass(function):
        function = function.__init__
    parameters = inspect.signature(function).parameters
    for param_name, param_obj in parameters.items():
        if param_obj.annotation is not inspect.Parameter.empty:
            arg_info = param_obj.annotation
            arg_info['default'] = param_obj.default
            arg_info['dest'] = param_name
            arg_info['type'] = arg_info.get('type', type(param_obj.default))

            if arg_info['type'] is bool:
                arg_info['type'] = str2bool
                arg_info['nargs'] = '?'
                arg_info['const'] = True

            if 'choices' in arg_info:
                arg_info['help'] = arg_info.get('help', '') + f" (choices: {', '.join(arg_info['choices'])})"
                arg_info['metavar'] = param_name.upper()

            options = {f'--{param_name}', f'--{param_name.replace("_", "-")}'}
            custom_options = arg_info.pop('option', [])
            custom_options = [custom_options] if isinstance(custom_options, str) else custom_options
            options.update(custom_options)
            options = sorted(sorted(list(options)), key=len)
            parser.add_argument(*options, **arg_info)


def strip_unexpected_kwargs(func, kwargs):
    signature = inspect.signature(func)
    parameters = signature.parameters

    # check if the function has kwargs
    for name, param in parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    kwargs = {arg: value for arg, value in kwargs.items() if arg in parameters}
    return kwargs


def from_args(func, ns, *args, **kwargs):
    return func(*args, **strip_unexpected_kwargs(func, vars(ns)), **kwargs)


def print_args(args):
    message = [f'{name}: {colored_text(str(value), TermColors.FG.cyan)}' for name, value in vars(args).items()]
    print(', '.join(message) + '\n')


def colored_text(msg, color):
    if isinstance(color, str):
        color = TermColors.FG.__dict__[color]
    return color.value + msg + TermColors.Control.reset.value


class Enum(enum.Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class EnumAction(Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        _enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if _enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(_enum, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in _enum))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = _enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        enum = self._enum(values)  # noqa
        setattr(namespace, self.dest, enum)


class TermColors:
    class Control(enum.Enum):
        reset = '\033[0m'
        bold = '\033[01m'
        disable = '\033[02m'
        underline = '\033[04m'
        reverse = '\033[07m'
        strikethrough = '\033[09m'
        invisible = '\033[08m'

    class FG(enum.Enum):
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class BG(enum.Enum):
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'
