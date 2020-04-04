import math

import torch
from torch.distributions import Bernoulli
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import train_test_split_edges, degree

from datasets import get_dataloader
from gnn import GCNConv, GConvMixedDP
from models import GCNClassifier, Node2VecClassifier, GCNLinkPredictor, Node2VecLinkPredictor

params = {
    'nodeclass': {
        'gcn': {
            'hidden_dim': 16,
            'epochs': 200,
            'optim': {
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
            },
            'epochs': 100,
            'optim': {
                'weight_decay': 0,
                'lr': 0.01
            },
        }
    },
    'linkpred': {
        'gcn': {
            'hidden_dim': 32,
            'output_dim': 16,
            'epochs': 200,
            'optim': {
                'weight_decay': 0,
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
            },
            'epochs': 100,
            'optim': {
                'weight_decay': 0,
                'lr': 0.01
            },
        }
    },
}


class Task:
    @staticmethod
    def task_name():
        raise NotImplementedError

    @staticmethod
    def one_bit_response(x, epsilon, alpha, delta, priv_dim=-1):
        if priv_dim == -1:
            priv_dim = x.size(1)
        exp = math.exp(epsilon)
        x_priv = x[:, :priv_dim]
        p = (x_priv - alpha[:priv_dim]) / delta[:priv_dim]
        p[torch.isnan(p)] = 0.  # nan happens when alpha = beta, so also data.x = alpha, so the prev fraction must be 0
        p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
        x_priv = Bernoulli(p).sample()
        x = torch.cat([x_priv, x[:, priv_dim:]], dim=1)
        return x

    def __init__(self, dataset, model_name, feature, epsilon, priv_dim):
        self.dataset = dataset
        self.model_name = model_name
        self.feature = feature
        self.epsilon = epsilon
        self.priv_dim = priv_dim if self.feature == 'priv' else 0

    @torch.no_grad()
    def prepare_data(self):
        data = self.dataset[0]
        if self.feature == 'priv':
            data.x = self.one_bit_response(data.x, self.epsilon, data.alpha, data.delta, self.priv_dim)
        elif self.feature == 'locd':
            num_nodes = data.num_nodes
            data.x = None
            data.num_nodes = num_nodes
            data = LocalDegreeProfile()(data)
        return data

    def init_model(self, data):
        raise NotImplementedError

    def run(self):
        device = torch.device('cuda')
        data = self.prepare_data().to(device)
        model = self.init_model(data).to(device)
        loader = get_dataloader(self.dataset.name, data)
        optimizer = torch.optim.Adam(model.parameters(), **params[self.task_name()][self.model_name]['optim'])
        model.train_model(loader, optimizer, epochs=params[self.task_name()][self.model_name]['epochs'])
        return model.evaluate(loader)


class NodeClassification(Task):
    @staticmethod
    def task_name():
        return 'nodeclass'

    def init_model(self, data):
        if self.model_name == 'gcn':
            model = GCNClassifier(
                input_dim=data.num_node_features,
                output_dim=self.dataset.num_classes,
                hidden_dim=params['nodeclass']['gcn']['hidden_dim'],
                priv_input_dim=self.priv_dim,
                epsilon=self.epsilon,
                alpha=data.alpha,
                delta=data.delta,
            )
        else:
            model = Node2VecClassifier(
                data.num_nodes,
                **params['nodeclass']['node2vec']['params']
            )
        return model


class LinkPrediction(Task):
    @staticmethod
    def task_name():
        return 'linkpred'

    def prepare_data(self):
        data = super().prepare_data()
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data)
        data.edge_index = data.train_pos_edge_index
        return data

    def init_model(self, data):
        if self.model_name == 'gcn':
            model = GCNLinkPredictor(
                input_dim=data.num_node_features,
                output_dim=params['linkpred']['gcn']['output_dim'],
                hidden_dim=params['linkpred']['gcn']['hidden_dim'],
                priv_input_dim=self.priv_dim,
                epsilon=self.epsilon,
                alpha=data.alpha,
                delta=data.delta,
            )
        else:
            model = Node2VecLinkPredictor(
                data.num_nodes,
                **params['linkpred']['node2vec']['params']
            )
        return model


class ErrorEstimation(Task):
    @staticmethod
    def task_name():
        return 'errorest'

    def __init__(self, dataset, model_name, feature, epsilon, priv_dim):
        assert model_name == 'gcn' and feature == 'priv'
        super().__init__(dataset, model_name, feature, epsilon, priv_dim)
        self.device = 'cuda'
        data = self.dataset[0].to(self.device)
        delta = data.delta.clone()
        delta[delta == 0] = 1  # avoid inf and nan
        self.delta = delta
        gcnconv = GCNConv().to(self.device)
        self.gc = gcnconv(data.x, data.edge_index)

    def init_model(self, data):
        return GConvMixedDP(
            priv_dim=self.priv_dim,
            epsilon=self.epsilon,
            alpha=data.alpha,
            delta=data.delta)

    @torch.no_grad()
    def run(self):
        data = self.prepare_data().to(self.device)
        model = self.init_model(data).to(self.device)
        gc_hat = model(data.x, data.edge_index)
        diff = (self.gc - gc_hat) / self.delta
        error = torch.norm(diff, p=1, dim=1) / diff.shape[1]
        deg = self.get_degree(data)
        return list(zip(error.cpu().numpy(), deg.cpu().numpy()))

    @staticmethod
    def get_degree(data):
        row, col = data.edge_index
        return degree(row, data.num_nodes)
