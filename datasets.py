import json
import os
import ssl
from functools import partial

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, ToSparseTensor
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix

from transforms import NodeSplit, Normalize


class Facebook100(InMemoryDataset):
    url = 'https://escience.rpi.edu/data/DA/fb100/'
    targets = ['status', 'gender', 'major', 'minor', 'housing', 'year']

    def __init__(self, root, name, target, transform=None, pre_transform=None):
        self.name = name
        self.target = target
        assert target in self.targets
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return self.name + '.mat'

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        download_url(f'{self.url}/{self.raw_file_names}', self.raw_dir)
        ssl._create_default_https_context = context

    def process(self):
        mat = loadmat(os.path.join(self.raw_dir, self.raw_file_names))
        features = pd.DataFrame(mat['local_info'][:, :-1], columns=self.targets)
        if self.target == 'year':
            features.loc[(features['year'] < 2004) | (features['year'] > 2009), 'year'] = 0
        y = torch.from_numpy(LabelEncoder().fit_transform(features[self.target]))
        if 0 in features[self.target].values:
            y = y - 1

        x = features.drop(columns=self.target).replace({0: pd.NA})
        x = torch.tensor(pd.get_dummies(x).values, dtype=torch.float)
        edge_index = from_scipy_sparse_matrix(mat['A'])[0]
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(y))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Facebook100-{self.name}()'


class Twitch(InMemoryDataset):
    url = 'http://snap.stanford.edu/data/twitch.zip'
    available_datasets = {'DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU'}

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'twitch', self.name)

    @property
    def raw_file_names(self):
        return [
            f'musae_{self.name}_edges.csv',
            f'musae_{self.name}'+('' if self.name == 'DE' else '_features') + '.json',
            f'musae_{self.name}_target.csv'
        ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'twitch', self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        file = download_url(self.url, self.root)
        extract_zip(file, self.root)
        os.unlink(file)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file, usecols=['mature'])
        y = torch.from_numpy(y.to_numpy(dtype=int)).squeeze()
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        features = json.load(open(feature_file))
        data = [[int(node), feature, 1.0] for node, items in features.items() for feature in items]
        df = pd.DataFrame(data, columns=['node_id', 'feature_id', 'value']).drop_duplicates()
        df = df.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = torch.from_numpy(df.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Twitch-{self.name}()'


class KarateClub(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'KarateClub-{self.name}()'


class Elliptic(InMemoryDataset):
    url = 'https://uofi.box.com/shared/static/vhmlkw9b24sxsfwh5in9jypmx2azgaac.zip'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return [
            os.path.join('elliptic_bitcoin_dataset', file) for file in
            ['elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_features.csv']
        ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def num_classes(self):
        return 2

    def download(self):
        file = download_url(self.url, self.raw_dir)
        extract_zip(file, self.raw_dir)
        os.unlink(file)

    def process(self):
        file_features = os.path.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_features.csv')
        df = pd.read_csv(file_features, index_col=0, header=None)
        x = torch.from_numpy(df.to_numpy()).float()

        file_classes = os.path.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_classes.csv')
        df = pd.read_csv(file_classes, index_col='txId', na_values='unknown').fillna(0) - 1
        y = torch.from_numpy(df.to_numpy()).view(-1).long()
        num_nodes = y.size(0)

        df_idx = df.reset_index().reset_index().drop(columns='class').set_index('txId')
        file_edges = os.path.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_edgelist.csv')
        df = pd.read_csv(file_edges).join(df_idx, on='txId1', how='inner')
        df = df.join(df_idx, on='txId2', how='inner', rsuffix='2').drop(columns=['txId1', 'txId2'])
        edge_index = torch.from_numpy(df.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Elliptic()'


class GraphDataModule(LightningDataModule):
    available_datasets = {
        'cora': partial(Planetoid, name='cora', split='full'),
        'citeseer': partial(Planetoid, name='citeseer', split='full'),
        'pubmed': partial(Planetoid, name='pubmed', split='full'),
        'facebook': partial(KarateClub, name='facebook', pre_transform=NodeSplit()),
        'github': partial(KarateClub, name='github', pre_transform=NodeSplit()),
        'lastfm': partial(KarateClub, name='lastfm', pre_transform=NodeSplit()),
    }

    def __init__(self, name, root='datasets', normalize=False, sparse=False, transform=None, device='cpu'):
        super().__init__()
        self.name = name
        self.dataset = self.available_datasets[name](root=os.path.join(root, name), transform=transform)
        self.device = 'cpu' if not torch.cuda.is_available() else device
        self.data_list = None

        if normalize:
            low, high = normalize
            self.add_transform(Normalize(low, high))

        if sparse:
            self.add_transform(ToSparseTensor())

    def prepare_data(self):
        assert self.data_list is None
        self.data_list = [data.to(self.device) for data in self.dataset]

    def add_transform(self, transform):
        if self.has_prepared_data:
            self.data_list = [transform(data) for data in self.data_list]
        else:
            current_transform = self.dataset.transform
            new_transform = transform if current_transform is None else Compose([current_transform, transform])
            self.dataset.transform = new_transform

    def train_dataloader(self):
        return DataLoader(self.data_list, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_list, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_list, pin_memory=True)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, item):
        if not self.has_prepared_data:
            self.prepare_data()
        return self.data_list[item]

    def __str__(self):
        return self.dataset.__str__()


def available_datasets():
    return list(GraphDataModule.available_datasets.keys())
