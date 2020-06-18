from abc import abstractmethod
from functools import partial

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv

from datasets import load_dataset
from privacy import privatize
from models import NodeClassifier, LinkPredictor
from params import get_params


class TrainOnlyProgressBar(ProgressBar):
    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.disable = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


class Task:
    NodeClassification = 'node'
    LinkPrediction = 'link'
    ErrorEstimation = 'error'

    def __init__(self, data, model_name):
        self.model_name = model_name
        self.data = data

    @abstractmethod
    def run(self, logger): pass

    @staticmethod
    def task_list():
        return [Task.NodeClassification, Task.LinkPrediction, Task.ErrorEstimation]


class LearningTask(Task):
    def __init__(self, task_name, data, model_name):
        assert task_name in LearningTask.task_list()
        super().__init__(data, model_name)
        self.task_name = task_name
        self.model = self.get_model()

    @staticmethod
    def task_list():
        return [Task.NodeClassification, Task.LinkPrediction]

    def get_model(self):
        Model = {
            ('node', 'gcn'): partial(NodeClassifier),
            ('link', 'gcn'): partial(LinkPredictor),
        }
        return Model[self.task_name, self.model_name](
            data=self.data,
            **get_params(
                section='model',
                task=self.task_name,
                dataset=self.data.name,
                model_name=self.model_name
            )
        )

    def run(self, logger):
        early_stop_callback = EarlyStopping(**get_params(
            section='early-stop',
            task=self.task_name,
            dataset=self.data.name,
            model_name=self.model_name
        ))

        trainer = Trainer(
            gpus=torch.cuda.is_available(),
            callbacks=[TrainOnlyProgressBar()],
            checkpoint_callback=False,
            logger=logger,
            row_log_interval=1000,
            log_save_interval=1000,
            weights_summary=None,
            deterministic=True,
            progress_bar_refresh_rate=5,
            early_stop_callback=early_stop_callback,
            **get_params(
                section='trainer',
                task=self.task_name,
                dataset=self.data.name,
                model_name=self.model_name
            )
        )
        trainer.fit(self.model)
        trainer.test()





def main():
    seed_everything(12345)
    dataset = load_dataset('flickr', min_degree=3).to('cuda')
    dataset = privatize(dataset, 'bit', pfr=1, eps=3)
    for i in range(1):
        task = LearningTask(task_name='node', data=dataset, model_name='gcn')
        task.run(False)


if __name__ == '__main__':
    main()
