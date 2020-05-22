from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import _logger as log


class CustomMLFlowLogger(MLFlowLogger):
    def __init__(self, experiment_name):
        super().__init__(experiment_name=experiment_name)
        self._expt_id = None

    def log_metrics(self, metrics, step=None):
        if 'test_result' in metrics:
            super().log_metrics(metrics, step)

    def log_params(self, params):
        for key, value in params.items():
            self.experiment.log_param(self.run_id, key, value)

    def create_run(self, tags=None):
        run = self._mlflow_client.create_run(experiment_id=self.experiment_id, tags=tags)
        self._run_id = run.info.run_id
        return self._run_id

    def delete_runs(self, filter_string):
        query_results = self.experiment.search_runs(experiment_ids=self.experiment_id, filter_string=filter_string)
        for result in query_results:
            self.experiment.delete_run(result.info.run_id)

    @property
    def experiment_id(self):
        if self._expt_id is None:
            expt = self.experiment.get_experiment_by_name(self.experiment_name)

            if expt:
                self._expt_id = expt.experiment_id
            else:
                log.warning(f'Experiment with name {self.experiment_name} not found. Creating it.')
                self._expt_id = self._mlflow_client.create_experiment(name=self.experiment_name)

        return self._expt_id

    @property
    def run_id(self):
        if self._run_id is None:
            self._run_id = self.create_run()
        return self._run_id
