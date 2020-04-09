import os
import sys
from abc import abstractmethod
from contextlib import contextmanager

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only
from torch_geometric.utils import degree

try: from tsnecuda import TSNE
except ImportError: from sklearn.manifold import TSNE

from datasets import load_dataset
from gnn import GConvDP
from models import GCNClassifier, Node2VecClassifier, GCNLinkPredictor, Node2VecLinkPredictor

params = {
    'nodeclass': {
        'gcn': {
            'params': {
                'hidden_dim': 16,
                'weight_decay': 5e-4,
                'lr': 0.01
            },
        },
        'node2vec': {
            'params': {
                'embedding_dim': 128,
                'walk_length': 20,
                'context_size': 10,
                'walks_per_node': 10,
                'batch_size': 128,
                'lr': 0.01
            },
        },
        'early_stop': {
            'min_delta': 0,
            'patience': 5
        }
    },
    'linkpred': {
        'gcn': {
            'params': {
                'hidden_dim': 64,
                'output_dim': 32,
                'dropout': 0,
                'lr': 0.01,
                'weight_decay': 0
            },
        },
        'node2vec': {
            'params': {
                'embedding_dim': 32,
                'walk_length': 20,
                'context_size': 10,
                'walks_per_node': 10,
                'batch_size': 128,
                'lr': 0.01,
                'weight_decay': 0
            },
        },
        'early_stop': {
            'min_delta': 0,
            'patience': 5
        }
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


class ResultLogger(LightningLoggerBase):
    @property
    def experiment(self):
        return self

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if 'test_result' in metrics:
            # noinspection PyAttributeOutsideInit
            self.result = metrics['test_result']

    @rank_zero_only
    def log_hyperparams(self, parameters): pass

    @property
    def name(self): return 'ResultLogger'

    @property
    def version(self): return 0.1


class Task:
    @staticmethod
    def task_name(): raise NotImplementedError

    def __init__(self, data, model_name, epsilon, **kwargs):
        self.model_name = model_name
        self.epsilon = epsilon
        self.data = data

    @abstractmethod
    def run(self): pass


class LearningTask(Task):

    @abstractmethod
    def get_model(self): pass

    def run(self, **train_args):
        logger = ResultLogger()
        model = self.get_model()
        trainer = Trainer(
            gpus=1, check_val_every_n_epoch=10, checkpoint_callback=False, logger=logger, **train_args
        )
        trainer.fit(model)
        with silence_stdout(): trainer.test()
        return logger.result


class NodeClassification(LearningTask):
    task_name = 'nodeclass'

    def get_model(self):
        if self.model_name == 'gcn':
            return GCNClassifier(
                data=self.data,
                epsilon=self.epsilon,
                **params['nodeclass']['gcn']['params']
            )
        else:
            return Node2VecClassifier(
                data=self.data,
                **params['nodeclass']['node2vec']['params']
            )

    def run(self, **train_args):
        monitor = 'val_loss' if self.model_name == 'gcn' else 'val_acc'
        early_stop_callback = EarlyStopping(monitor=monitor, mode='auto', **params['nodeclass']['early_stop'])
        return super().run(early_stop_callback=early_stop_callback, **train_args)


class LinkPrediction(LearningTask):
    task_name = 'linkpred'

    def get_model(self):
        if self.model_name == 'gcn':
            return GCNLinkPredictor(
                data=self.data,
                epsilon=self.epsilon,
                **params['linkpred']['gcn']['params']
            )
        else:
            return Node2VecLinkPredictor(
                data=self.data,
                **params['linkpred']['node2vec']['params']
            )

    def run(self, **train_args):
        early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', **params['linkpred']['early_stop'])
        return super().run(early_stop_callback=early_stop_callback, **train_args)


class ErrorEstimation(Task):
    task_name = 'errorest'

    def __init__(self, data, orig_features, model_name, epsilon):
        super().__init__(data, model_name, epsilon)
        self.model = GConvDP(epsilon=self.epsilon, alpha=data.alpha, delta=data.delta)
        self.gc = self.model(orig_features, data.edge_index, False)

    @torch.no_grad()
    def run(self, **kwargs):
        gc_hat = self.model(self.data.x, self.data.edge_index, self.data.priv_mask)
        diff = (self.gc - gc_hat) / self.data.delta
        error = torch.norm(diff, p=1, dim=1) / diff.shape[1]
        deg = self.get_degree(self.data)
        return list(zip(error.cpu().numpy(), deg.cpu().numpy()))

    @staticmethod
    def get_degree(data):
        row, col = data.edge_index
        return degree(row, data.num_nodes)


# class VisualizeEmbedding(LinkPrediction):
#     @staticmethod
#     def task_name():
#         return 'visual'
#
#     def __init__(self, dataset, model_name, feature, epsilon, priv_dim):
#         assert model_name == 'gcn' and feature == 'priv'
#         super().__init__(dataset, model_name, feature, epsilon, priv_dim)
#
#     def run(self, **train_args):
#         model = self.get_model()
#         early_stop_callback = EarlyStopping(monitor='val_auc', mode='max', **params['linkpred']['early_stop'])
#         trainer = Trainer(early_stop_callback=early_stop_callback, **train_args)
#         trainer.fit(model)
#         z = model(self.dataset[0])
#         return TSNE(n_components=2).fit_transform(z.cpu().numpy())


def main():
    torch.manual_seed(12345)
    data = load_dataset(
        dataset_name='cora',
        # task_name='linkpred'
    ).to('cuda')
    for i in range(1):
        result = NodeClassification(
            data, model_name='gcn', feature='priv', epsilon=2, priv_dim=data.num_node_features//2
        ).run(max_epochs=100)
        print(result)


if __name__ == '__main__':
    main()
