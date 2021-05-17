import os
from functools import partial

import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.data.utils import download

from transforms import Normalize, FilterTopClass, NodeSplit


class KarateClub(DGLDataset):
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, raw_dir, name, transform=None):
        self._g = None
        self.transform = transform
        name = name.lower()
        assert name in self.available_datasets
        url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
        super().__init__(name=name, url=url, raw_dir=raw_dir, force_reload=False, verbose=True)

    def download(self):
        for part in ['edges', 'features', 'target']:
            download(f'{self.url}/{self.name}/{part}.csv', os.path.join(self.raw_dir, part + '.csv'))

    def process(self):
        target_file = os.path.join(self.raw_dir, 'target.csv')
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, 'edges.csv')
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()

        feature_file = os.path.join(self.raw_dir, 'features.csv')
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()

        g = dgl.graph(data=(edge_index[0], edge_index[1]), num_nodes=num_nodes)
        g = dgl.add_reverse_edges(g)
        g.ndata['feat'] = x
        g.ndata['label'] = y

        if self.transform:
            g = self.transform(g)

        self._g = g

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    def save(self):
        dgl.save_graphs(os.path.join(self.raw_dir, self.name + '.bin'), self._g)

    def load(self):
        self._g = dgl.load_graphs(os.path.join(self.raw_dir, self.name + '.bin'))[0][0]

    def has_cache(self):
        return os.path.exists(os.path.join(self.raw_dir, self.name + '.bin'))


supported_datasets = {
    'cora': CoraGraphDataset,
    'pubmed': PubmedGraphDataset,
    'facebook': partial(KarateClub, name='facebook'),
    'lastfm': partial(KarateClub, name='lastfm', transform=FilterTopClass(10)),
}


def load_dataset(
        dataset:        dict(help='name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir:       dict(help='directory to store the dataset') = './datasets',
        data_range:     dict(help='min and max feature value', nargs=2, type=float) = (0, 1),
        val_ratio:      dict(help='fraction of nodes used for validation') = .25,
        test_ratio:     dict(help='fraction of nodes used for test') = .25,
        ):
    g = supported_datasets[dataset](raw_dir=os.path.join(data_dir, dataset))[0]
    g = NodeSplit(val_ratio=val_ratio, test_ratio=test_ratio)(g)
    g.name = dataset
    g.num_features = g.ndata['feat'].size(1)
    g.num_classes = int(g.ndata['label'].max().item()) + 1

    if data_range is not None:
        low, high = data_range
        g = Normalize(low, high)(g)

    return g
