import pandas as pd
import torch
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm, trange

from datasets import load_dataset, get_dataloader
from models import GCNClassifier, Node2VecClassifier, GCNLinkPredictor, Node2VecLinkPredictor
from utils import convert_data

torch.manual_seed(12345)

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
    def __init__(self, dataset, model_name, feature, epsilon, priv_dim):
        device = torch.device('cuda')
        self.dataset = dataset
        self.model_name = model_name
        self.feature = feature
        self.epsilon = epsilon
        self.priv_dim = priv_dim if self.feature == 'priv' else 0
        self.data = self.prepare_data().to(device)
        self.model = self.init_model().to(device)

    def prepare_data(self):
        data = self.dataset[0]
        return convert_data(data, self.feature, priv_dim=self.priv_dim, epsilon=self.epsilon)

    def init_model(self):
        raise NotImplementedError

    def run(self, epochs, **optimargs):
        loader = get_dataloader(self.dataset.name, self.data)
        optimizer = torch.optim.Adam(self.model.parameters(), **optimargs)
        self.model.train_model(loader, optimizer, epochs)
        return self.model.evaluate(loader)


class NodeClassification(Task):
    task_name = 'nodeclass'

    def init_model(self):
        if self.model_name == 'gcn':
            model = GCNClassifier(
                input_dim=self.data.num_node_features,
                output_dim=self.dataset.num_classes,
                hidden_dim=params['nodeclass']['gcn']['hidden_dim'],
                priv_input_dim=self.priv_dim,
                epsilon=self.epsilon,
                alpha=self.data.alpha[:self.priv_dim],
                delta=self.data.delta[:self.priv_dim],
            )
        else:
            model = Node2VecClassifier(
                self.data.num_nodes,
                **params['nodeclass']['node2vec']['params']
            )
        return model

    def __str__(self):
        return 'nodeclass'


class LinkPrediction(Task):
    task_name = 'linkpred'

    def prepare_data(self):
        data = super().prepare_data()
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data)
        return data

    def init_model(self):
        if self.model_name == 'gcn':
            model = GCNLinkPredictor(
                input_dim=self.data.num_node_features,
                output_dim=params['linkpred']['gcn']['output_dim'],
                hidden_dim=params['linkpred']['gcn']['hidden_dim'],
                priv_input_dim=self.priv_dim,
                epsilon=self.epsilon,
                alpha=self.data.alpha[:self.priv_dim],
                delta=self.data.delta[:self.priv_dim],
            )
        else:
            model = Node2VecLinkPredictor(
                self.data.num_nodes,
                **params['linkpred']['node2vec']['params']
            )
        return model

    def __str__(self):
        return 'linkpred'


def experiment():
    tasks = [NodeClassification, LinkPrediction]
    datasets = ['cora', 'citeseer', 'pubmed']
    models = ['gcn', 'node2vec']
    features = {
        'gcn': ['raw', 'priv', 'locd'],
        'node2vec': ['dummy']
    }
    epsilons = [0.1, 1, 3, 5, 7, 9]
    epsilons_pr = [1, 3, 5]
    private_ratios = [0.1, 0.2, 0.50, 1]
    repeats = 10

    for task in tqdm(tasks, desc='task'):
        for dataset_name in tqdm(datasets, desc=f'(task={task.task_name}) dataset', leave=False):
            dataset = load_dataset(dataset_name)
            for model in tqdm(models, desc=f'(dataset={dataset_name}) model', leave=False):
                results = []
                for feature in tqdm(features[model], desc=f'(model={model}) feature', leave=False):
                    pr_list = private_ratios if feature == 'priv' else [0]
                    for private_ratio in tqdm(pr_list, desc=f'(feature={feature}) private ratio', leave=False):
                        if feature == 'priv':
                            eps_list = epsilons if private_ratio == 1 else epsilons_pr
                        else:
                            eps_list = [0]
                        for epsilon in tqdm(eps_list, desc=f'(ratio={private_ratio}) epsilon', leave=False):
                            for run in trange(repeats, desc=f'(epsilon={epsilon}) run', leave=False):
                                performance = task(
                                    dataset, model, feature, epsilon,
                                    priv_dim=int(private_ratio * dataset.num_node_features)
                                ).run(
                                    epochs=params[task.task_name][model]['epochs'],
                                    **params[task.task_name][model]['optim']
                                )
                                results.append((run, f'{model}+{feature}', epsilon, performance))

                df_result = pd.DataFrame(data=results, columns=['run', 'conf', 'eps', 'perf'])
                df_result.to_pickle(f'results/{task.task_name}_{dataset_name}_{model}.pkl')


if __name__ == '__main__':
    experiment()
