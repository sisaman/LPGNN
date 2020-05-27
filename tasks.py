from abc import abstractmethod
from functools import partial

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv

from datasets import load_dataset
from mechanisms import privatize
from models import GCNClassifier, VGAELinkPredictor
from params import get_params


class LitProgressBar(ProgressBar):
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
        early_stop_callback = EarlyStopping(**get_params(
            section='early-stop',
            task=self.task_name,
            dataset=self.data.name,
            model_name=self.model_name
        ))

        trainer = Trainer(
            gpus=1,
            callbacks=[LitProgressBar()],
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


class ErrorEstimation(Task):
    def __init__(self, data, orig_features):
        super().__init__(data, 'gcn')
        self.model = GCNConv(data.num_features, data.num_features, cached=True).cuda()
        self.model.weight.data.copy_(torch.eye(data.num_features))  # no linear transformation
        self.gc = self.model(orig_features, data.edge_index)
        alpha = orig_features.min(dim=0)[0]
        beta = orig_features.max(dim=0)[0]
        self.delta = beta - alpha

    @torch.no_grad()
    def run(self, logger):
        gc_hat = self.model(self.data.x, self.data.edge_index)
        diff = (self.gc - gc_hat) / self.delta
        diff[:, (self.delta == 0)] = 0  # eliminate division by zero
        error = torch.norm(diff, p=1, dim=1) / self.data.num_features
        deg = self.get_degree(self.data)
        logger.log_metrics({'test_result': list(zip(error.cpu().numpy(), deg.cpu().numpy()))})

    @staticmethod
    def get_degree(data):
        row, col = data.edge_index
        return degree(row, data.num_nodes)


def main():
    seed_everything(12345)
    dataset = load_dataset('pubmed').to('cuda')
    dataset = privatize(dataset, 'bit', pfr=0.1, eps=3)
    for i in range(1):
        task = LearningTask(task_name='node', data=dataset, model_name='gcn')
        task.run(False)


if __name__ == '__main__':
    main()
