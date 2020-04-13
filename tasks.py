import os
import sys
import warnings

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

params = {
    'nodeclass': {
        'gcn': {
            'params': {
                'hidden_dim': 16,
            },
            'optim': {
                'cora': {
                    'lr': 0.01,
                    'weight_decay': 0.01
                },
                'citeseer': {
                    'lr': 0.01,
                    'weight_decay': 0.1
                },
                'pubmed': {
                    'lr': 0.01,
                    'weight_decay': 0.001
                },
                'flickr': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
                'amazon-photo': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
                'amazon-computers': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
            }
        },
        'node2vec': {
            'params': {
                'embedding_dim': 128,
                'walk_length': 80,
                'context_size': 10,
                'walks_per_node': 10,
                'batch_size': 1
            },
        }
    },
    'linkpred': {
        'vgae': {
            'params': {
                'output_dim': 16,
            },
            'optim': {
                'cora': {
                    'lr': 0.01,
                    'weight_decay': 0.001
                },
                'citeseer': {
                    'lr': 0.01,
                    'weight_decay': 0.01
                },
                'pubmed': {
                    'lr': 0.1,
                    'weight_decay': 0.0001
                },
                'flickr': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
                'amazon-photo': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
                'amazon-computers': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
            }
        },
        'node2vec': {
            'params': {
                'embedding_dim': 128,
                'walk_length': 80,
                'context_size': 10,
                'walks_per_node': 10,
                'batch_size': 1
            },
        },
    }
}


@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


class Task:
    @staticmethod
    def task_name(): raise NotImplementedError

    def __init__(self, data, model_name, epsilon=1, **kwargs):
        self.model_name = model_name
        self.epsilon = epsilon
        self.data = data

    @abstractmethod
    def run(self): pass


class LearningTask(Task):
    def __init__(self, data, model_name, epsilon=1, **kwargs):
        super().__init__(data, model_name, epsilon, **kwargs)
        self.trained_model = None

    @abstractmethod
    def get_model(self): pass

    def run(self, **train_args):
        logger = ResultLogger()
        model = self.get_model()
        early_stop_callback = None if self.model_name == 'node2vec' else EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10
        )
        # noinspection PyTypeChecker
        trainer = Trainer(
            gpus=1, checkpoint_callback=False, logger=logger, weights_summary=None,
            early_stop_callback=early_stop_callback, **train_args
        )
        trainer.fit(model)
        with silence_stdout(): trainer.test()
        self.trained_model = model
        return logger.result


class NodeClassification(LearningTask):
    task_name = 'nodeclass'

    def get_model(self):
        if self.model_name == 'gcn':
            # noinspection PyArgumentList
            return GCNClassifier(
                data=self.data,
                epsilon=self.epsilon,
                **params[self.task_name][self.model_name]['params'],
                **params[self.task_name][self.model_name]['optim'][self.data.name],
            )
        elif self.model_name == 'node2vec':
            return Node2VecClassifier(
                data=self.data,
                **params[self.task_name][self.model_name]['params']
            )


class LinkPrediction(LearningTask):
    task_name = 'linkpred'

    def get_model(self):
        if self.model_name == 'vgae':
            # noinspection PyArgumentList
            return VGAELinkPredictor(
                data=self.data,
                epsilon=self.epsilon,
                **params[self.task_name][self.model_name]['params'],
                **params[self.task_name][self.model_name]['optim'][self.data.name],
            )
        elif self.model_name == 'node2vec':
            return Node2VecLinkPredictor(
                data=self.data,
                **params[self.task_name][self.model_name]['params']
            )


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


class Visualization(LinkPrediction):
    task_name = 'visualize'

    def get_model(self):
        return VGAELinkPredictor(
            data=self.data,
            epsilon=self.epsilon,
            **params['linkpred'][self.model_name]['params'],
            **params['linkpred'][self.model_name]['optim'][self.data.name],
        )

    def run(self, **train_args):
        super().run(**train_args)
        z = self.trained_model(self.data)
        embedding = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
        label = self.data.y.cpu().numpy()
        return {'data': embedding, 'label': label}


if __name__ == '__main__':
    dataset = load_dataset('cora', split_edges=True).to('cuda')
    eps = 5
    dataset = privatize(dataset, pnr=1, pfr=1, eps=eps, method='lap')
    dataset = NormalizeFeatures()(dataset)
    task = NodeClassification(dataset, 'gcn', epsilon=eps)
    print(task.run(min_epochs=10, max_epochs=500))
