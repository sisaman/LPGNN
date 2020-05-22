import argparse
from typing import Union, Dict, Optional, Any
from pytorch_lightning.loggers import LightningLoggerBase, MLFlowLogger


class CustomMLFlowLogger(LightningLoggerBase):
    def __init__(self, experiment_name):
        super().__init__()
        self.logger = MLFlowLogger(experiment_name=experiment_name)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key in metrics:
            if key.startswith('test_'):
                self.logger.log_metrics(metrics, step)
                break

    def log_hyperparams(self, params: argparse.Namespace):
        self.logger.log_hyperparams(params)

    def finalize(self, status: str):
        self.logger.finalize(status)

    @property
    def name(self) -> str:
        return self.logger.name

    @property
    def version(self) -> Union[int, str]:
        return self.logger.version

    @property
    def experiment(self) -> Any:
        return self.logger.experiment
