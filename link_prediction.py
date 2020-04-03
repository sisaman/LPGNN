import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import negative_sampling, add_remaining_self_loops, train_test_split_edges
from tqdm import tqdm, trange

from datasets import load_dataset
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
            },
            'epochs': 100,
            'batch_size': 128,
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


class LinkPredictor(torch.nn.Module):
    @staticmethod
    def get_link_labels(pos_edge_index, neg_edge_index):
        link_labels = torch.zeros(pos_edge_index.size(1) +
                                  neg_edge_index.size(1), device=pos_edge_index.device).float()
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        raise NotImplementedError

    def train_model(self, data, optimizer, epochs):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, data):
        self.eval()
        pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        link_logits = self.get_link_logits(data.x, pos_edge_index, neg_edge_index)
        link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = torch.sigmoid(link_logits)
        link_probs = link_probs.cpu().numpy()
        link_labels = link_labels.cpu().numpy()

        return roc_auc_score(link_labels, link_probs)


class Node2VecLinkPredictor(Node2Vec, LinkPredictor):
    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = self(total_edge_index[0])
        x_i = self(total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)

    def train_model(self, data, optimizer, epochs):
        self.train()
        for epoch in trange(epochs, desc='Epoch', leave=False):
            nodes = torch.arange(data.num_nodes, device=data.train_pos_edge_index.device)
            node_sampler = DataLoader(nodes, batch_size=setup['model']['node2vec']['batch_size'], shuffle=True)
            for subset in node_sampler:
                optimizer.zero_grad()
                loss = self.loss(data.train_pos_edge_index, subset)
                loss.backward()
                optimizer.step()


class GCNLinkPredictor(GCN, LinkPredictor):

    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        x = self(x, pos_edge_index)
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)

    def train_model(self, data, optimizer, epochs):
        self.train()
        for epoch in trange(epochs, desc='Epoch', leave=False):
            optimizer.zero_grad()
            x, pos_edge_index = data.x, data.train_pos_edge_index
            pos_edge_index_with_self_loops, _ = add_remaining_self_loops(pos_edge_index, num_nodes=x.size(0))

            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                num_neg_samples=pos_edge_index.size(1))

            link_logits = self.get_link_logits(x, pos_edge_index, neg_edge_index)
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            loss = binary_cross_entropy_with_logits(link_logits, link_labels)
            loss.backward()
            optimizer.step()


def link_prediction(dataset, model_name, feature, epsilon):
    device = torch.device('cuda')
    data = convert_data(dataset[0], feature, epsilon=epsilon)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    data = data.to(device)

    if model_name == 'gcn':
        model = GCNLinkPredictor(
            input_dim=data.num_node_features,
            output_dim=setup['model']['gcn']['output_dim'],
            hidden_dim=setup['model']['gcn']['hidden_dim'],
            priv_input_dim=(data.num_node_features if feature == 'priv' else 0),
            epsilon=epsilon,
            alpha=data.alpha,
            delta=data.delta,
        )
    else:
        model = Node2VecLinkPredictor(
            data.num_nodes,
            **setup['model']['node2vec']['params']
        )

    # loader = get_dataloader(dataset.name, data)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), **setup['model'][model_name]['optim'])
    model.train_model(data, optimizer, setup['model'][model_name]['epochs'])
    return model.evaluate(data)


def experiment():
    for dataset_name in tqdm(setup['datasets'], desc='Dataset'):
        results = []
        dataset = load_dataset(dataset_name)
        for run in trange(setup['repeats'], desc='Run', leave=False):
            for model in tqdm(setup['model'], desc='Model', leave=False):
                for feature in setup['model'][model].get('feature', ['']):
                    auc = -1
                    for epsilon in tqdm(setup['eps'], desc='Epsilon', leave=False):
                        if feature == 'priv' or auc == -1:
                            auc = link_prediction(dataset, model, feature, epsilon)
                        results.append((run, f'{model}+{feature}', epsilon, auc))

        df_result = pd.DataFrame(data=results, columns=['run', 'conf', 'eps', 'auc'])
        df_result.to_pickle(f'results/link_prediction_{dataset_name}.pkl')


if __name__ == '__main__':
    experiment()
