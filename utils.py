import os
import pandas as pd
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.loggers import LightningLoggerBase


class TrainOnlyProgressBar(ProgressBar):
    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.disable = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    # Todo: to be removed in pytorch-lightning version > 1.8.1
    @property
    def total_val_batches(self) -> int:
        trainer = self.trainer
        total_val_batches = 0
        if trainer.fast_dev_run and trainer.val_dataloaders is not None:
            total_val_batches = len(trainer.val_dataloaders)
        elif not self.trainer.disable_validation:
            is_val_epoch = trainer.current_epoch % trainer.check_val_every_n_epoch == 0
            total_val_batches = trainer.num_val_batches if is_val_epoch else [0]
            total_val_batches = sum(total_val_batches)
        return total_val_batches


class PandasLogger(LightningLoggerBase):

    def __init__(self, experiment_name, output_dir, write_mode='replace'):
        assert write_mode in ['replace', 'truncate']
        super().__init__()
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)
        self.filename = os.path.join(output_dir, self.experiment_name + '.pkl')
        self.save_mode = write_mode
        self.data = []
        self.metrics = {}
        self.params = None
        self.df = None

    def log_metrics(self, metrics, step=None):
        if 'test_result' in metrics:
            self.metrics.update(metrics)
        pass

    def log_params(self, params):
        self._add_run_to_data()
        self.params = params
        pass

    def log_hyperparams(self, params):
        pass

    def dump(self):
        print('\nSaving results...', end='')
        params = self.params.keys()
        self._add_run_to_data()
        df_new = pd.DataFrame(self.data)
        self.data = []

        self.df = self._open_or_create_dataframe()
        self.df = self.df.append(df_new, ignore_index=True)

        if self.save_mode == 'replace':
            self.df.drop_duplicates(subset=params, keep='last', inplace=True, ignore_index=True)

        self.df.to_pickle(path=self.filename)
        print('done!\n')

    def _add_run_to_data(self):
        if self.params is not None:
            run = {**self.params, **self.metrics}
            self.data.append(run)
            self.metrics = {}
            self.params = None

    def _open_or_create_dataframe(self):
        self.df = pd.DataFrame()
        if self.save_mode == 'replace':
            try:
                self.df = pd.read_pickle(self.filename)
            except FileNotFoundError:
                pass
        return self.df

    @property
    def experiment(self):
        return self.data

    @property
    def name(self):
        return self.experiment_name

    @property
    def version(self):
        return 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dump()


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
