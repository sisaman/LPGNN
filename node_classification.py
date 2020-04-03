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
        'cora',
        'citeseer',
        'pubmed',
        'flickr',
        # 'reddit',
        # 'ppi',
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
            'batch_size': 128,
            'optim': {
                'weight_decay': 0,
                'lr': 0.01
            },
        }
    },
    'eps': {
        'general': [0.1, 1, 3, 5, 7, 9],
        'ratio': [1, 3, 5]
    },
    'private_ratio': [0, 0.1, 0.2, 0.50, 1],
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


def node_classification(dataset, model_name, feature, epsilon, priv_dim):
    device = torch.device('cuda')
    data = dataset[0].to(device)
    data = convert_data(data, feature, priv_dim=priv_dim, epsilon=epsilon)

    if model_name == 'gcn':
        model = GCNClassifier(
            input_dim=data.num_node_features,
            output_dim=dataset.num_classes,
            hidden_dim=setup['model']['gcn']['hidden_dim'],
            priv_input_dim=(priv_dim if feature == 'priv' else 0),
            epsilon=epsilon,
            alpha=data.alpha[:priv_dim],
            delta=data.delta[:priv_dim],
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
    print('experiment: general')
    for dataset_name in tqdm(setup['datasets'], desc='Dataset'):
        results = []
        dataset = load_dataset(dataset_name)
        for model in tqdm(setup['model'], desc='Model', leave=False):
            for run in trange(setup['repeats'], desc='Run', leave=False):
                for feature in setup['model'][model].get('feature', ['']):
                    acc = -1
                    for epsilon in tqdm(setup['eps']['general'], desc='Epsilon', leave=False):
                        if feature == 'priv' or acc == -1:
                            acc = node_classification(dataset, model, feature, epsilon,
                                                      priv_dim=dataset.num_node_features)
                        results.append((run, f'{model}+{feature}', epsilon, acc))

            df_result = pd.DataFrame(data=results, columns=['run', 'conf', 'eps', 'acc'])
            df_result.to_pickle(f'results/node_classification_{dataset_name}_{model}.pkl')


def private_ratio_experiment():
    print('experiment: ratio')
    for dataset_name in tqdm(setup['datasets'], desc='Dataset'):
        results = []
        dataset = load_dataset(dataset_name)
        for run in trange(setup['repeats'], desc='Run', leave=False):
            for private_ratio in tqdm(setup['private_ratio'], desc='Ratio', leave=False):
                for epsilon in tqdm(setup['eps']['ratio'], desc='Epsilon', leave=False):
                    acc = node_classification(dataset, 'gcn', 'priv', epsilon,
                                              priv_dim=int(private_ratio * dataset.num_node_features))
                    results.append((run, private_ratio, epsilon, acc))

        df_result = pd.DataFrame(data=results, columns=['run', 'pr', 'eps', 'acc'])
        df_result.to_pickle(f'results/node_classification_pr_{dataset_name}.pkl')


def test():
    dataset = load_dataset('cora')
    print('node features:', dataset.num_node_features)
    # priv_dim = int(0.1 * dataset.num_node_features)
    priv_dim = 5
    print('priv dim:', priv_dim)
    acc = node_classification(dataset, 'gcn', 'priv', 0.1, priv_dim=priv_dim)
    print('accuracy:', acc)


if __name__ == '__main__':
    # private_ratio_experiment()
    experiment()
    # test()
