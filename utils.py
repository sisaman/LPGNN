import os
import torch
import numpy as np
import pandas as pd
import random
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, save_dir, **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        num_files = len(os.listdir(save_dir))
        self.save_dir = os.path.join(save_dir, str(num_files))
        self.writer = SummaryWriter(log_dir=self.save_dir, **kwargs)

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.writer.add_scalars(k, v, step)
            else:
                self.writer.add_scalar(k, v, step)

    def save(self):
        self.writer.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_args(args):
    args = {key: str(value) for key, value in vars(args).items()}
    df_args = pd.DataFrame.from_dict(args, orient='index')
    print(tabulate(df_args, tablefmt='fancy_grid'), '\n')


def colored_text(msg, color):
    if isinstance(color, str):
        color = TermColors.FG.__dict__[color]
    return color + msg + TermColors.reset


class TermColors:
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class FG:
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

    class BG:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'
