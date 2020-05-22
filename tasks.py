import os
import sys
from abc import abstractmethod
from contextlib import contextmanager
from functools import partial

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.utils import degree

from datasets import load_dataset

try:
    from tsnecuda import TSNE
except ImportError:
    from sklearn.manifold import TSNE

from gnn import GConv
from models import GCNClassifier, VGAELinkPredictor
from params import get_params


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
            ('node', 'gcn'): partial(GCNClassifier),
            ('link', 'gcn'): partial(VGAELinkPredictor),
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
        # logger = ResultLogger()
        # logger = CustomMLFlowLogger(experiment_name='dpgcn')
        # logger = MLFlowLogger(experiment_name='dpgcn')
        early_stop_callback = EarlyStopping(**get_params(
            section='early-stop',
            task=self.task_name,
            dataset=self.data.name,
            model_name=self.model_name
        ))

        trainer = Trainer(
            gpus=1,
            checkpoint_callback=False,
            logger=logger,
            row_log_interval=500,
            log_save_interval=500,
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
        # with self.silence_stdout(): trainer.test()
        # return logger.result
        trainer.test()

    @contextmanager
    def silence_stdout(self):
        new_target = open(os.devnull, "w")
        old_target = sys.stdout
        sys.stdout = new_target
        try:
            yield new_target
        finally:
            sys.stdout = old_target


class ErrorEstimation(Task):
    def __init__(self, data, orig_features):
        super().__init__(data, 'gcn')
        self.model = GConv(cached=False)
        self.gc = self.model(orig_features, data.edge_index)

    @torch.no_grad()
    def run(self, logger):
        gc_hat = self.model(self.data.x, self.data.edge_index)
        diff = (self.gc - gc_hat)  # / self.data.delta
        # diff[:, (self.data.delta == 0)] = 0  # eliminate division by zero
        error = torch.norm(diff, p=1, dim=1) / diff.shape[1]
        deg = self.get_degree(self.data)
        return list(zip(error.cpu().numpy(), deg.cpu().numpy()))

    @staticmethod
    def get_degree(data):
        row, col = data.edge_index
        return degree(row, data.num_nodes)


def main():
    seed_everything()
    dataset = load_dataset('cora', split_edges=True).to('cuda')
    task = LearningTask(task_name='link', data=dataset, model_name='gcn')
    print(task.run(False))


if __name__ == '__main__':
    main()
