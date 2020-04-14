import os
import sys
import warnings
from functools import partial

from torch_geometric.transforms import NormalizeFeatures

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from abc import abstractmethod
from contextlib import contextmanager

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.utils import degree

from datasets import load_dataset, privatize

try: from tsnecuda import TSNE
except ImportError: from sklearn.manifold import TSNE

from gnn import GConvDP
from models import GCNClassifier, Node2VecClassifier, Node2VecLinkPredictor, VGAELinkPredictor, ResultLogger
from params import get_params


class Task:
    task_name = None

    def __init__(self, data, model_name, epsilon=1):
        self.model_name = model_name
        self.epsilon = epsilon
        self.data = data

    @abstractmethod
    def run(self): pass


class LearningTask(Task):
    def __init__(self, task_name, data, model_name, epsilon=1):
        super().__init__(data, model_name, epsilon)
        self.task_name = task_name
        self.model = self.get_model(epsilon)

    def get_model(self, epsilon):
        Model = {
            ('nodeclass', 'gcn'): partial(GCNClassifier, epsilon=epsilon),
            ('nodeclass', 'node2vec'): Node2VecClassifier,
            ('linkpred', 'gcn'): partial(VGAELinkPredictor, epsilon=epsilon),
            ('linkpred', 'node2vec'): Node2VecLinkPredictor,
        }
        return Model[self.task_name, self.model_name](
            data=self.data,
            **get_params(
                section='model', task=self.task_name, dataset=self.data.name, model=self.model_name
            )
        )

    def run(self):
        logger = ResultLogger()
        early_stop_callback = None if self.model_name == 'node2vec' else EarlyStopping(**get_params(
            section='early-stop', task=self.task_name, dataset=self.data.name, model=self.model_name
        ))
        # noinspection PyTypeChecker
        trainer = Trainer(
            gpus=1, checkpoint_callback=False, logger=logger, weights_summary=None,
            early_stop_callback=early_stop_callback, **get_params(
                section='trainer', task=self.task_name, dataset=self.data.name, model=self.model_name
            )
        )
        trainer.fit(self.model)
        with self.silence_stdout(): trainer.test()
        return logger.result

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
    task_name = 'errorest'

    def __init__(self, data, orig_features, model_name, epsilon):
        super().__init__(data, model_name, epsilon)
        self.model = GConvDP(epsilon=self.epsilon, alpha=data.alpha, delta=data.delta, cached=False)
        self.gc = self.model(orig_features, data.edge_index, False)

    @torch.no_grad()
    def run(self, **kwargs):
        gc_hat = self.model(self.data.x, self.data.edge_index, self.data.priv_mask)
        diff = (self.gc - gc_hat) / self.data.delta
        diff[:, (self.data.delta == 0)] = 0  # eliminate division by zero
        error = torch.norm(diff, p=1, dim=1) / diff.shape[1]
        deg = self.get_degree(self.data)
        return list(zip(error.cpu().numpy(), deg.cpu().numpy()))

    @staticmethod
    def get_degree(data):
        row, col = data.edge_index
        return degree(row, data.num_nodes)


class Visualization(LearningTask):
    def __init__(self, data, epsilon=1):
        super().__init__(task_name='linkpred', data=data, model_name='gcn', epsilon=epsilon)

    def run(self):
        super().run()
        z = self.model(self.data)
        embedding = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
        label = self.data.y.cpu().numpy()
        return {'data': embedding, 'label': label}


def main():
    dataset = load_dataset('cora', split_edges=True).to('cuda')
    task = LearningTask(task_name='linkpred', data=dataset, model_name='node2vec', epsilon=1)
    print(task.run())


if __name__ == '__main__':
    main()
