import numpy as np
import torch
import torch.nn.functional as F
# from torch_geometric.utils import subgraph
from mechanisms import supported_feature_mechanisms, RandomizedResopnse


class FeatureTransform:
    supported_features = ['raw', 'rnd', 'one', 'ohd']

    def __init__(self, feature: dict(help='feature transformation method',
                                     choices=supported_features, option='-f') = 'raw'):

        self.feature = feature

    def __call__(self, g):

        if self.feature == 'rnd':
            g.ndata['feat'] = torch.rand_like(g.ndata['feat'])
        elif self.feature == 'ohd':
            g = OneHotDegree(max_degree=g.num_features - 1)(g)
        elif self.feature == 'one':
            g.ndata['feat'] = torch.ones_like(g.ndata['feat'])

        return g


class FeaturePerturbation:
    def __init__(self,
                 mechanism:     dict(help='feature perturbation mechanism', choices=list(supported_feature_mechanisms),
                                     option='-m') = 'mbm',
                 x_eps:         dict(help='privacy budget for feature perturbation', type=float,
                                     option='-ex') = np.inf,
                 data_range=None):

        self.mechanism = mechanism
        self.input_range = data_range
        self.x_eps = x_eps

    def __call__(self, g):
        if np.isinf(self.x_eps):
            return g

        if self.input_range is None:
            self.input_range = g.ndata['feat'].min().item(), g.ndata['feat'].max().item()

        g.ndata['feat'] = supported_feature_mechanisms[self.mechanism](
            eps=self.x_eps,
            input_range=self.input_range
        )(g.ndata['feat'])

        return g


class LabelPerturbation:
    def __init__(self,
                 y_eps: dict(help='privacy budget for label perturbation',
                             type=float, option='-ey') = np.inf):
        self.y_eps = y_eps

    def __call__(self, g):
        g.ndata['label'] = F.one_hot(g.ndata['label'], num_classes=g.num_classes)
        p_ii = 1  # probability of preserving the clean label i
        p_ij = 0  # probability of perturbing label i into another label j

        if not np.isinf(self.y_eps):
            mechanism = RandomizedResopnse(eps=self.y_eps, d=g.num_classes)
            perturb_mask = g.ndata['train_mask'] | g.ndata['val_mask']
            y_perturbed = mechanism(g.ndata['label'][perturb_mask])
            g.ndata['label'][perturb_mask] = y_perturbed
            p_ii, p_ij = mechanism.p, mechanism.q

        # set label transistion matrix
        g.T = torch.ones(g.num_classes, g.num_classes, device=g.device) * p_ij
        g.T.fill_diagonal_(p_ii)

        return g


class OneHotDegree:
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, g):
        degree = g.in_degrees()
        degree.clamp_(max=self.max_degree)
        g.ndata['feat'] = F.one_hot(degree, num_classes=self.max_degree + 1).float()  # add 1 for zero degree
        return g


class Normalize:
    def __init__(self, low, high):
        self.min = low
        self.max = high

    def __call__(self, g):
        x = g.ndata['feat']
        alpha = x.min(dim=0)[0]
        beta = x.max(dim=0)[0]
        delta = beta - alpha
        x = (x - alpha) * (self.max - self.min) / delta + self.min
        x = x[:, torch.nonzero(delta, as_tuple=False).squeeze()]  # remove features with delta = 0
        g.ndata['feat'] = x
        g.num_features = x.size(1)
        return g


class FilterTopClass:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        y = torch.nn.functional.one_hot(data.y)
        c = y.sum(dim=0).sort(descending=True)
        y = y[:, c.indices[:self.num_classes]]
        idx = y.sum(dim=1).bool()

        data.x = data.x[idx]
        data.y = y[idx].argmax(dim=1)
        data.num_nodes = data.y.size(0)

        if 'adj_t' in data:
            data.adj_t = data.adj_t[idx, idx]
        elif 'edge_index' in data:
            data.edge_index, data.edge_attr = subgraph(idx, data.edge_index, data.edge_attr, relabel_nodes=True)

        if 'train_mask' in data:
            data.train_mask = data.train_mask[idx]
            data.val_mask = data.val_mask[idx]
            data.test_mask = data.test_mask[idx]

        return data


class NodeSplit:
    def __init__(self, val_ratio=.25, test_ratio=.25):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def __call__(self, g):
        n_val = int(self.val_ratio * g.num_nodes())
        n_test = int(self.test_ratio * g.num_nodes())
        perm = torch.randperm(g.num_nodes())
        val_nodes = perm[:n_val]
        test_nodes = perm[n_val:n_val + n_test]
        train_nodes = perm[n_val + n_test:]
        val_mask = torch.zeros_like(g.ndata['val_mask'], dtype=torch.bool)
        val_mask[val_nodes] = True
        test_mask = torch.zeros_like(g.ndata['test_mask'], dtype=torch.bool)
        test_mask[test_nodes] = True
        train_mask = torch.zeros_like(g.ndata['train_mask'], dtype=torch.bool)
        train_mask[train_nodes] = True
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        g.ndata['train_mask'] = train_mask
        return g
