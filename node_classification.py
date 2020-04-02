import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from tqdm import tqdm, trange

from datasets import load_dataset, get_dataloader
from gnn import GCN
from utils import convert_data

torch.manual_seed(12345)

setup = {
    'datasets': [
        # 'cora',
        'citeseer',
        # 'pubmed',
        # 'reddit',
        # 'ppi',
        # 'flickr',
        # 'yelp',
    ],
    'model': {
        'gcn': {
            'feature': [
                'raw',
                'priv',
                'locd'
            ],
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
            },
            'epochs': 100,
            'batch_size': 512,
            'optim': {
                'weight_decay': 0,
                'lr': 0.01
            },
        }
    },
    'eps': [
        0.1,
        1,
        3,
        5,
        7,
        9,
    ],
    'repeats': 10,
}


class GCNClassifier(GCN):
    def train_model(self, dataloader, optimizer, epochs):
        self.train()
        for epoch in trange(epochs, desc='Epoch', leave=False):
            for batch in dataloader:
                if batch.train_mask.any():
                    optimizer.zero_grad()
                    out = self(batch.x, batch.edge_index)
                    loss = cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                    loss.backward()
                    optimizer.step()

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        total_nodes, total_corrects = 0, 0
        for batch in dataloader:
            pred = self(batch.x, batch.edge_index).argmax(dim=1)
            total_corrects += (pred[batch.test_mask] == batch.y[batch.test_mask]).sum().item()
            total_nodes += batch.test_mask.sum().item()

        acc = total_corrects / total_nodes
        return acc


class Node2VecClassifier(Node2Vec):
    def train_model(self, dataloader, optimizer, epochs):
        self.train()
        for epoch in trange(epochs, desc='Epoch', leave=False):
            for data in dataloader:
                nodes = torch.arange(data.num_nodes, device=data.edge_index.device)
                node_sampler = DataLoader(nodes, batch_size=setup['model']['node2vec']['batch_size'], shuffle=True)
                for subset in node_sampler:
                    optimizer.zero_grad()
                    loss = self.loss(data.edge_index, subset)
                    loss.backward()
                    optimizer.step()

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        total_nodes, total_corrects = 0, 0
        for data in dataloader:
            z = self(torch.arange(data.num_nodes, device=data.edge_index.device))
            acc = self.test(z[data.train_mask], data.y[data.train_mask],
                            z[data.test_mask], data.y[data.test_mask], max_iter=150)
            n_nodes = data.test_mask.sum().item()
            total_corrects += acc * n_nodes
            total_nodes += n_nodes
        return total_corrects / total_nodes


def node_classification(dataset, model_name, feature, epsilon):
    device = torch.device('cuda')
    data = dataset[0].to(device)
    data = convert_data(data, feature, epsilon=epsilon)

    if model_name == 'gcn':
        model = GCNClassifier(
            input_dim=data.num_node_features,
            output_dim=dataset.num_classes,
            hidden_dim=setup['model']['gcn']['hidden_dim'],
            priv_input_dim=(data.num_node_features if feature == 'priv' else 0),
            epsilon=epsilon,
            alpha=data.alpha,
            delta=data.delta,
        )
    else:
        model = Node2VecClassifier(
            data.num_nodes,
            **setup['model']['node2vec']['params']
        )

    loader = get_dataloader(dataset.name, data)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), **setup['model'][model_name]['optim'])
    model.train_model(loader, optimizer, setup['model'][model_name]['epochs'])
    return model.evaluate(loader)


def experiment():
    for dataset_name in tqdm(setup['datasets'], desc='Dataset'):
        results = []
        dataset = load_dataset(dataset_name)
        for run in trange(setup['repeats'], desc='Run', leave=False):
            for model in tqdm(setup['model'], desc='Model', leave=False):
                for feature in setup['model'][model].get('feature', ['']):
                    acc = -1
                    for epsilon in tqdm(setup['eps'], desc='Epsilon', leave=False):
                        if feature == 'priv' or acc == -1:
                            acc = node_classification(dataset, model, feature, epsilon)
                        results.append((run, f'{model}+{feature}', epsilon, acc))

        df_result = pd.DataFrame(data=results, columns=['run', 'conf', 'eps', 'acc'])
        df_result.to_pickle(f'results/node_classification_{dataset_name}.pkl')


if __name__ == '__main__':
    experiment()
