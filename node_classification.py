import pandas as pd
import torch
from torch.nn.functional import log_softmax, nll_loss
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import LocalDegreeProfile
from tqdm import tqdm, trange

from datasets import load_dataset, get_dataloader
from gnn import GCN
from utils import one_bit_response

torch.manual_seed(12345)

setup = {
    'datasets': [
        'cora',
        # 'citeseer',
        # 'pubmed',
        # 'reddit',
        # 'ppi',
        # 'flickr',
        # 'yelp',
    ],
    'methods': [
        'private',
        'node2vec',
        'default',
        'localdegree',
        # 'random',
    ],
    'eps': [
        0.1,
        # 0.2,
        # 0.5,
        1,
        3,
        5,
        7,
        9,
    ],
    'hidden_dim': 16,
    'epochs': 100,
    'repeats': 10,
}


class GCNClassifier(GCN):

    def forward(self, data):
        x = super().forward(data)
        return log_softmax(x, dim=1)

    def train_model(self, dataloader, optimizer, epochs):
        self.train()
        for epoch in trange(epochs, desc='Epoch', leave=False):
            for batch in dataloader:
                if batch.train_mask.any():
                    optimizer.zero_grad()
                    out = self(batch)
                    loss = nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
                    loss.backward()
                    optimizer.step()

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        total_nodes, total_corrects = 0, 0
        for batch in dataloader:
            pred = self(batch).argmax(dim=1)
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
                node_sampler = DataLoader(nodes, batch_size=128, shuffle=True)
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


def run(dataset, method, epsilon):
    device = torch.device('cuda')
    data = dataset[0].to(device)

    with torch.no_grad():

        if method == 'random':
            data.x = torch.rand(data.num_nodes, data.num_node_features, device=device) * data.delta + data.alpha
        elif method == 'localdegree':
            data.x = None
            data.num_nodes = len(data.y)
            data = LocalDegreeProfile()(data)
        elif method.startswith('private'):
            data = one_bit_response(data, epsilon)

        if method == 'node2vec':
            model = Node2VecClassifier(
                data.num_nodes, embedding_dim=16, walk_length=20, context_size=10, walks_per_node=10
            )
        else:
            model = GCNClassifier(
                input_dim=data.num_node_features,
                output_dim=dataset.num_classes,
                hidden_dim=setup['hidden_dim'],
                private=(method.startswith('private')),
                epsilon=epsilon,
                alpha=data.alpha,
                delta=data.delta,
            )

        loader = get_dataloader(dataset.name, data)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train_model(loader, optimizer, 100)
    return model.evaluate(loader)


def node_classification():
    for dataset_name in tqdm(setup['datasets'], desc='Dataset'):
        results = []
        dataset = load_dataset(dataset_name)
        for run_counter in trange(setup['repeats'], desc='Run', leave=False):
            for method in tqdm(setup['methods'], desc='Method', leave=False):
                for epsilon in tqdm(setup['eps'] if method.startswith('private') else [0], desc='Epsilon',
                                    leave=False):
                    acc = run(dataset, method, epsilon)
                    if not method.startswith('private'):
                        for eps in setup['eps']:
                            results.append((run_counter, method, eps, acc))
                    else:
                        results.append((run_counter, method, epsilon, acc))

        df_result = pd.DataFrame(data=results, columns=['run', 'conf', 'eps', 'acc'])
        df_result.to_pickle(f'results/node_classification_{dataset_name}.pkl')


if __name__ == '__main__':
    node_classification()
