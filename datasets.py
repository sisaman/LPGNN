import os
import os.path as osp
from functools import partial

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip, DataLoader
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import GDC, Compose
from torch_geometric.utils import to_undirected, from_networkx
from karateclub.dataset import GraphReader
from transforms import NodeSplit, Normalize, EdgeSplit


class KarateClub(GraphReader, InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name.lower()
        super().__init__(dataset=name)
        super(InMemoryDataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                              pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        data = from_networkx(self.get_graph())
        data.x = torch.tensor(self.get_features().todense(), dtype=torch.float)
        print(data.x.size())
        data.y = torch.tensor(self.get_target())

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


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
    'twitch': partial(KarateClub, name='twitch', pre_transform=NodeSplit()),
    'flickr': Flickr,
    'elliptic': Elliptic,
}


def get_available_datasets():
    return list(available_datasets.keys())


class GraphDataset(LightningDataModule):
    def __init__(self, dataset_name, data_dir='datasets', normalize=True,
                 split_edges=False, use_gdc=False, device='cpu'):
        super().__init__()
        self.name = dataset_name
        self.root_dir = os.path.join(data_dir, dataset_name)

        transforms = [
            # FeatureSelection(100)
        ]
        if normalize:
            transforms.append(Normalize())
        if split_edges:
            transforms.append(EdgeSplit(val_ratio=0.1, test_ratio=0.1))
        if use_gdc:
            transforms.append(
                GDC(self_loop_weight=1, normalization_in='sym', normalization_out='sym',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05, eps=1e-4),
                    # diffusion_kwargs=dict(method='heat', t=10),
                    # sparsification_kwargs=dict(method='topk', k=256, dim=0),
                    sparsification_kwargs=dict(method='threshold', avg_degree=256),
                    exact=False)
            )

        self.transforms = Compose(transforms)
        self.use_gdc = use_gdc
        self.device = device
        self.data = None

    def prepare_data(self):
        assert self.data is None
        dataset = available_datasets[self.name](root=self.root_dir, transform=self.transforms)
        self.num_classes = dataset.num_classes
        self.data = dataset[0]

        if self.device == 'cuda' and torch.cuda.is_available():
            self.data.to('cuda')

    def apply_transform(self, transform):
        if not self.has_prepared_data:
            self.prepare_data()
        self.data = transform(self.data)

    def get_data(self):
        if not self.has_prepared_data:
            self.prepare_data()
        return self.data

    def train_dataloader(self):
        return DataLoader([self.data], pin_memory=True)

    def val_dataloader(self):
        return DataLoader([self.data], pin_memory=True)

    def test_dataloader(self):
        return DataLoader([self.data], pin_memory=True)

    @property
    def num_features(self):
        return self.data.num_features
