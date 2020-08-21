import math
import os
import os.path as osp
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip, DataLoader
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import GDC, Compose
from torch_geometric.utils import to_undirected, negative_sampling


class MUSAE(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    # noinspection DuplicatedCode
    def process(self):
        data_list = self.read_musae()

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def read_musae(self):
        filenames = os.listdir(self.raw_dir)
        raw_files = sorted([osp.join(self.raw_dir, f) for f in filenames])
        x, edge_index, y, num_nodes = None, None, None, None

        for file in raw_files:
            if 'target' in file:
                y = pd.read_csv(file)['target']
                y = torch.from_numpy(y.to_numpy(dtype=np.int))
                num_nodes = y.size(0)
            elif 'edges' in file:
                edge_index = pd.read_csv(file)
                edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
                edge_index = to_undirected(edge_index, num_nodes)  # undirected edges
            elif 'features' in file:
                x = pd.read_csv(file).drop_duplicates()
                x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
                x = torch.from_numpy(x.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        seed = sum([ord(c) for c in 'musae'])
        rng = torch.Generator().manual_seed(seed)
        data = NodeSplit(rng=rng)(data)
        return [data]

    def __repr__(self):
        return 'MUSAE-{}({})'.format(self.name, len(self))


class Elliptic(InMemoryDataset):
    url = 'https://uofi.box.com/shared/static/vhmlkw9b24sxsfwh5in9jypmx2azgaac.zip'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return [
            osp.join('elliptic_bitcoin_dataset', file) for file in
            ['elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_features.csv']
        ]

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

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

    # noinspection DuplicatedCode
    def process(self):
        data_list = self.read_elliptic()

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def read_elliptic(self):
        file_features = osp.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_features.csv')
        df = pd.read_csv(file_features, index_col=0, header=None)
        x = torch.from_numpy(df.to_numpy()).float()

        file_classes = osp.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_classes.csv')
        df = pd.read_csv(file_classes, index_col='txId', na_values='unknown') - 1
        y = torch.from_numpy(df.to_numpy()).view(-1).float()
        num_nodes = y.size(0)

        df_idx = df.reset_index().reset_index().drop(columns='class').set_index('txId')
        file_edges = osp.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_edgelist.csv')
        df = pd.read_csv(file_edges).join(df_idx, on='txId1', how='inner')
        df = df.join(df_idx, on='txId2', how='inner', rsuffix='2').drop(columns=['txId1', 'txId2'])
        edge_index = torch.from_numpy(df.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        nodes_with_class = ~torch.isnan(y)
        num_nodes_with_class = nodes_with_class.sum().item()

        data = Data(num_nodes=num_nodes_with_class)
        seed = sum([ord(c) for c in 'bitcoin'])
        rng = torch.Generator().manual_seed(seed)
        data = NodeSplit(rng=rng)(data)

        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)

        val_mask[nodes_with_class] = data.val_mask
        test_mask[nodes_with_class] = data.test_mask
        train_mask[nodes_with_class] = data.train_mask

        data.x = x
        data.y = y.long()
        data.edge_index = edge_index
        data.num_nodes = num_nodes
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.train_mask = train_mask

        return [data]

    def __repr__(self):
        return f'Elliptic({len(self)})'


available_datasets = {
    'cora': partial(Planetoid, name='cora'),
    'citeseer': partial(Planetoid, name='citeseer'),
    'twitch': partial(MUSAE, name='twitch'),
    'flickr': Flickr,
    'elliptic': Elliptic,
}


def get_available_datasets():
    return list(available_datasets.keys())


class NodeSplit:
    def __init__(self, val_ratio=.25, test_ratio=.25, rng=None):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = rng

    def __call__(self, data):
        n_val = int(self.val_ratio * data.num_nodes)
        n_test = int(self.test_ratio * data.num_nodes)
        perm = torch.randperm(data.num_nodes, generator=self.rng)
        val_nodes = perm[:n_val]
        test_nodes = perm[n_val:n_val + n_test]
        train_nodes = perm[n_val + n_test:]
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask[val_nodes] = True
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[test_nodes] = True
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_nodes] = True
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.train_mask = train_mask
        return data


class EdgeSplit:
    def __init__(self, val_ratio=0.1, test_ratio=0.1, rng=None):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = rng

    def __call__(self, data):
        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(self.val_ratio * row.size(0)))
        n_t = int(math.floor(self.test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0), generator=self.rng)
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        neg_edge_index = negative_sampling(
            edge_index=torch.stack([row, col], dim=0),
            num_nodes=data.num_nodes,
            num_neg_samples=n_v + n_t
        )

        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:]

        return data


class Normalize:
    def __call__(self, data):
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        delta = beta - alpha
        data.x = (data.x - alpha) / delta
        data.x = data.x[:, torch.nonzero(delta, as_tuple=False).squeeze()]  # remove features with delta = 0
        return data


class GraphDataset(LightningDataModule):
    def __init__(self, dataset_name, data_dir='datasets', normalize=True, split_edges=False, use_gdc=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(data_dir, dataset_name)

        transforms = []
        if normalize:
            transforms.append(Normalize())
        if split_edges:
            transforms.append(EdgeSplit(val_ratio=0.1, test_ratio=0.1))
        if use_gdc:
            transforms.append(
                GDC(self_loop_weight=1, normalization_in='sym', normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=256, dim=0), exact=True)
            )

        self.transforms = Compose(transforms)
        self.use_gdc = use_gdc
        self.data = None

    def prepare_data(self):
        assert self.data is None
        dataset = available_datasets[self.dataset_name](root=self.root_dir, transform=self.transforms)
        self.num_classes = dataset.num_classes
        self.data = dataset[0]

    def apply_transform(self, transform):
        if not self.has_prepared_data:
            self.prepare_data()
        self.data = transform(self.data)

    def train_dataloader(self):
        return DataLoader([self.data], pin_memory=True)

    def val_dataloader(self):
        return DataLoader([self.data], pin_memory=True)

    def test_dataloader(self):
        return DataLoader([self.data], pin_memory=True)

    @property
    def num_features(self):
        return self.data.num_features
