import os
import time
from argparse import ArgumentParser

import inspect
import enum
import functools
import torch
import numpy as np
import pandas as pd
import random
from tabulate import tabulate
try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger:
    def __init__(self, project=None, name=None, config=None, save_code=True,
                 reinit=True, enabled=True, **kwargs):
        self.enabled = enabled
        if enabled:
            if wandb is None:
                raise ImportError('wandb is not installed yet, install it with `pip install wandb`.')

            os.environ["WANDB_SILENT"] = "true"

            self.experiment = wandb.init(
                project=project, name=name, config=config, save_code=save_code, reinit=reinit, **kwargs
            )

    def log(self, metrics, step=None):
        if self.enabled:
            self.experiment.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self.experiment.finish()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def measure_runtime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        print('\nTotal time spent:', end - start, 'seconds.\n\n')
        return out
    return wrapper


def add_parameters_as_argument(function, parser: ArgumentParser):
    parameters = inspect.signature(function).parameters
    for param_name, param_obj in parameters.items():
        if param_obj.annotation is not inspect.Parameter.empty:
            arg_info = param_obj.annotation
            arg_info['default'] = param_obj.default

            if 'action' not in arg_info:
                arg_info['type'] = arg_info.get('type', type(param_obj.default))

            if 'choices' in arg_info:
                arg_info['help'] = arg_info.get('help', '') + f" (choices: { ', '.join(arg_info['choices']) })"
                arg_info['metavar'] = param_name.upper()

            option = arg_info.pop('option', [f'--{param_name.replace("_", "-")}'])
            option = [option] if isinstance(option, str) else option
            parser.add_argument(*option, **arg_info)


def print_args(args):
    message = [f'{name}: {colored_text(str(value), TermColors.FG.cyan)}' for name, value in vars(args).items()]
    print(', '.join(message) + '\n')


def colored_text(msg, color):
    if isinstance(color, str):
        color = TermColors.FG.__dict__[color]
    return color.value + msg + TermColors.Control.reset.value


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
