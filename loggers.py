import os
import pandas as pd
from pytorch_lightning.loggers import LightningLoggerBase


class PandasLogger(LightningLoggerBase):

    def __init__(self, experiment_name, save_dir, save_mode='replace'):
        assert save_mode in ['replace', 'truncate']
        super().__init__()
        self.experiment_name = experiment_name
        self.df_dir = os.path.join(save_dir, self.experiment_name + '.pkl')
        self.save_mode = save_mode
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
        print('\n\nDUMP\n\n')
        params = self.params.keys()
        self._add_run_to_data()
        df_new = pd.DataFrame(self.data)
        self.data = []

        self.df = self._open_or_create_dataframe()
        self.df = self.df.append(df_new, ignore_index=True)

        if self.save_mode == 'replace':
            self.df.drop_duplicates(subset=params, keep='last', inplace=True, ignore_index=True)

        self.df.to_pickle(path=self.df_dir)

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
                self.df = pd.read_pickle(self.df_dir)
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
