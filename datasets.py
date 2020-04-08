import json
import math
import os.path as osp
from functools import partial

import numpy as np
import scipy.sparse as sp
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid, Reddit, PPI, SNAPDataset
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_undirected


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
    num_nodes = data.num_nodes
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = torch.randperm(neg_row.size(0), generator=rng, dtype=torch.long)[:min(n_v + n_t, neg_row.size(0))]
    # perm = random.sample(range(neg_row.size(0)),
    #                      min(n_v + n_t, neg_row.size(0)))
    # perm = torch.tensor(perm)
    # perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


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


class DataRange:
    def __call__(self, data):
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        delta = beta - alpha
        data.alpha = alpha
        data.beta = beta
        data.delta = delta
        return data


class EdgeSplit:
    def __init__(self, val_ratio=0.05, test_ratio=0.1, random_state=None):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def __call__(self, data):
        rng = torch.Generator().manual_seed(self.random_state)
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data, self.val_ratio, self.test_ratio, rng=rng)
        data.edge_index = data.train_pos_edge_index
        return data


class GraphLoader:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx == 0:
            return self.data
        else:
            raise IndexError


datasets = {
    'cora': partial(Planetoid, root='datasets/Planetoid', name='cora'),
    'citeseer': partial(Planetoid, root='datasets/Planetoid', name='citeseer'),
    'pubmed': partial(Planetoid, root='datasets/Planetoid', name='pubmed'),
    'reddit': partial(Reddit, root='datasets/Reddit'),
    'ppi': partial(PPI, root='datasets/PPI'),
    'flickr': partial(Flickr, root='datasets/Flickr'),
    'yelp': partial(Yelp, root='datasets/Yelp'),
    'facebook': partial(SNAPDataset, root='datasets/SNAP', name='ego-facebook')
}


def get_availabel_datasets():
    return list(datasets.keys())


def load_dataset(dataset_name, task_name=None):
    transforms = [DataRange()]
    if task_name == 'linkpred':
        seed = sum([ord(c) for c in dataset_name])
        transforms += [EdgeSplit(random_state=seed)]
    dataset = datasets[dataset_name](transform=Compose(transforms))
    data = dataset[0]
    data.name = dataset_name
    data.num_classes = dataset.num_classes
    return data
