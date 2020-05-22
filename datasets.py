import os
import json
import math
import os.path as osp
from functools import partial

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import to_undirected, negative_sampling


# noinspection DuplicatedCode
def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1, rng=None):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)
        rng: random number generator

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0), generator=rng)
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    # num_nodes = data.num_nodes
    # neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
    # neg_adj_mask = neg_adj_mask.triu(diagonal=1)
    # neg_adj_mask[row, col] = 0
    #
    # neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=True)
    # perm = torch.randperm(neg_row.size(0), generator=rng, dtype=torch.long)[:min(n_v + n_t, neg_row.size(0))]
    # neg_row, neg_col = neg_row[perm], neg_col[perm]
    #
    # neg_adj_mask[neg_row, neg_col] = 0
    # data.train_neg_adj_mask = neg_adj_mask

    # newly added code by Sina
    neg_edge_index = negative_sampling(
        edge_index=torch.stack([row, col], dim=0),
        num_nodes=data.num_nodes,
        num_neg_samples=n_v + n_t
    )

    data.val_neg_edge_index = neg_edge_index[:, :n_v]
    data.test_neg_edge_index = neg_edge_index[:, n_v:]

    return data


# noinspection DuplicatedCode
class Flickr(InMemoryDataset):
    r"""The Flickr dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing descriptions and common properties of images.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
    feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
    class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
    role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'

    def __init__(self, root, transform=None, pre_transform=None):
        super(Flickr, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = osp.join(self.raw_dir, 'adj_full.npz')
        gdd.download_file_from_google_drive(self.adj_full_id, path)

        path = osp.join(self.raw_dir, 'feats.npy')
        gdd.download_file_from_google_drive(self.feats_id, path)

        path = osp.join(self.raw_dir, 'class_map.json')
        gdd.download_file_from_google_drive(self.class_map_id, path)

        path = osp.join(self.raw_dir, 'role.json')
        gdd.download_file_from_google_drive(self.role_id, path)

    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


# noinspection DuplicatedCode
class Yelp(InMemoryDataset):
    r"""The Yelp dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing customer reviewers and their friendship.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    adj_full_id = '1Juwx8HtDwSzmVIJ31ooVa1WljI4U5JnA'
    feats_id = '1Zy6BZH_zLEjKlEFSduKE5tV9qqA_8VtM'
    class_map_id = '1VUcBGr0T0-klqerjAjxRmAqFuld_SMWU'
    role_id = '1NI5pa5Chpd-52eSmLW60OnB3WS5ikxq_'

    def __init__(self, root, transform=None, pre_transform=None):
        super(Yelp, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = osp.join(self.raw_dir, 'adj_full.npz')
        gdd.download_file_from_google_drive(self.adj_full_id, path)

        path = osp.join(self.raw_dir, 'feats.npy')
        gdd.download_file_from_google_drive(self.feats_id, path)

        path = osp.join(self.raw_dir, 'class_map.json')
        gdd.download_file_from_google_drive(self.class_map_id, path)

        path = osp.join(self.raw_dir, 'role.json')
        gdd.download_file_from_google_drive(self.role_id, path)

    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class MUSAE(InMemoryDataset):
    r"""A variety of graph datasets collected from MUSAE
    <https://arxiv.org/pdf/1909.13021.pdf>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

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

        return [Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)]

    def __repr__(self):
        return 'MUSAE-{}({})'.format(self.name, len(self))


class Elliptic(InMemoryDataset):
    """
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://uofi.box.com/shared/static/vhmlkw9b24sxsfwh5in9jypmx2azgaac.zip'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw', 'elliptic_bitcoin_dataset')

    @property
    def raw_file_names(self):
        return ['elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_features.csv']

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
        file_features = osp.join(self.raw_dir, 'elliptic_txs_features.csv')
        df = pd.read_csv(file_features, index_col=0, header=None)
        x = torch.from_numpy(df.to_numpy()).float()

        file_classes = osp.join(self.raw_dir, 'elliptic_txs_classes.csv')
        df = pd.read_csv(file_classes, index_col='txId', na_values='unknown') - 1
        y = torch.from_numpy(df.to_numpy()).view(-1).float()
        num_nodes = y.size(0)

        df_idx = df.reset_index().reset_index().drop(columns='class').set_index('txId')
        file_edges = osp.join(self.raw_dir, 'elliptic_txs_edgelist.csv')
        df = pd.read_csv(file_edges).join(df_idx, on='txId1', how='inner')
        df = df.join(df_idx, on='txId2', how='inner', rsuffix='2').drop(columns=['txId1', 'txId2'])
        edge_index = torch.from_numpy(df.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        nodes_with_class = ~torch.isnan(y)
        num_nodes_with_class = nodes_with_class.sum().item()

        data = Data(num_nodes=num_nodes_with_class)
        seed = sum([ord(c) for c in 'bitcoin'])
        rng = torch.Generator().manual_seed(seed)
        # noinspection PyTypeChecker
        data = train_test_split_nodes(data, rng=rng)

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
        return f'Elliptic-Bitcoin({len(self)})'


class GraphLoader:
    def __init__(self, data):
        self.data = data
        self.data.to = self.to

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx == 0:
            return self.data
        else:
            raise IndexError

    # TO BE REMOVED WHEN THE NEXT VERSION OF PYG IS RELEASED
    def to(self, device, *keys, **kwargs):
        return self.data.apply(lambda x: x.to(device, **kwargs), *keys)


def train_test_split_nodes(data, val_ratio=.25, test_ratio=.25, rng=None):
    n_val = int(val_ratio * data.num_nodes)
    n_test = int(test_ratio * data.num_nodes)
    perm = torch.randperm(data.num_nodes, generator=rng)
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


def stack_dataset(dataset):
    num_nodes = 0
    x = torch.empty(0, dtype=dataset[0].x.dtype)
    y = torch.empty(0, dtype=dataset[0].y.dtype)
    edge_index = torch.empty(0, dtype=dataset[0].edge_index.dtype)

    for data in dataset:
        x = torch.cat([x, data.x], dim=0)
        if data.y.size(0) == 1:
            data.y = data.y.repeat(data.x.size(0))
        y = torch.cat([y, data.y], dim=0)
        edge_index = torch.cat([edge_index, data.edge_index + num_nodes], dim=1)
        num_nodes += data.num_nodes

    return Data(x=x, y=y, edge_index=edge_index)


datasets = {
    'cora': partial(Planetoid, root='datasets/Planetoid', name='cora'),
    'citeseer': partial(Planetoid, root='datasets/Planetoid', name='citeseer'),
    'pubmed': partial(Planetoid, root='datasets/Planetoid', name='pubmed'),
    'coauthor-cs': partial(Coauthor, root='datasets/Coauthor/cs', name='cs'),
    'coauthor-ph': partial(Coauthor, root='datasets/Coauthor/ph', name='physics'),
    'flickr': partial(Flickr, root='datasets/Flickr'),
    'amazon-photo': partial(Amazon, root='datasets/Amazon/photo', name='photo'),
    'amazon-computers': partial(Amazon, root='datasets/Amazon/computers', name='computers'),
    'facebook': partial(MUSAE, root='datasets/MUSAE', name='facebook'),
    'github': partial(MUSAE, root='datasets/MUSAE', name='github'),
    'twitch': partial(MUSAE, root='datasets/MUSAE', name='twitch'),
    'bitcoin': partial(Elliptic, root='datasets/Elliptic')
}


def get_available_datasets():
    return list(datasets.keys())


def load_dataset(dataset_name, split_edges=False):
    dataset = datasets[dataset_name]()
    assert len(dataset) == 1

    data = dataset[0]
    data.name = dataset_name
    data.num_classes = dataset.num_classes
    seed = sum([ord(c) for c in dataset_name])
    rng = torch.Generator().manual_seed(seed)

    if split_edges:
        data.train_mask = data.val_mask = data.test_mask = None
        data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1, rng=rng)
        data.edge_index = data.train_pos_edge_index
    elif not hasattr(data, 'train_mask'):
        data = train_test_split_nodes(data, val_ratio=.25, test_ratio=.25, rng=rng)

    # normalize features between zero and one
    alpha = data.x.min(dim=0)[0]
    beta = data.x.max(dim=0)[0]
    delta = beta - alpha
    data.x = (data.x - alpha) / delta
    data.x[:, (delta == 0)] = 0

    # shuffle x columns
    data.x = data.x[:, torch.randperm(data.num_features)]

    return data


if __name__ == '__main__':
    load_dataset('bitcoin', False)
