import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from mechanisms import supported_feature_mechanisms, RandomizedResopnse


class FeatureTransform:
    supported_features = ['raw', 'rnd', 'one', 'ohd']

    def __init__(self, feature: dict(help='feature transformation method',
                                     choices=supported_features, option='-f') = 'raw'):

        self.feature = feature

    def __call__(self, data):

        if self.feature == 'rnd':
            data.x = torch.rand_like(data.x)
        elif self.feature == 'ohd':
            data = OneHotDegree(max_degree=data.num_features - 1)(data)
        elif self.feature == 'one':
            data.x = torch.ones_like(data.x)

        return data


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

    def __call__(self, data):
        if np.isinf(self.x_eps):
            return data

        if self.input_range is None:
            self.input_range = data.x.min().item(), data.x.max().item()

        data.x = supported_feature_mechanisms[self.mechanism](
            eps=self.x_eps,
            input_range=self.input_range
        )(data.x)

        return data


class LabelPerturbation:
    def __init__(self,
                 y_eps: dict(help='privacy budget for label perturbation',
                             type=float, option='-ey') = np.inf):
        self.y_eps = y_eps

    def __call__(self, data):
        data.y = F.one_hot(data.y, num_classes=data.num_classes)
        p_ii = 1  # probability of preserving the clean label i
        p_ij = 0  # probability of perturbing label i into another label j

        if not np.isinf(self.y_eps):
            mechanism = RandomizedResopnse(eps=self.y_eps, d=data.num_classes)
            perturb_mask = data.train_mask | data.val_mask
            y_perturbed = mechanism(data.y[perturb_mask])
            data.y[perturb_mask] = y_perturbed
            p_ii, p_ij = mechanism.p, mechanism.q

        # set label transistion matrix
        data.T = torch.ones(data.num_classes, data.num_classes, device=data.y.device) * p_ij
        data.T.fill_diagonal_(p_ii)

        return data


class OneHotDegree:
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, data):
        degree = data.adj_t.sum(dim=0).long()
        degree.clamp_(max=self.max_degree)
        data.x = F.one_hot(degree, num_classes=self.max_degree + 1).float()  # add 1 for zero degree
        return data


class Normalize:
    def __init__(self, low, high):
        self.min = low
        self.max = high

    def __call__(self, data):
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        delta = beta - alpha
        data.x = (data.x - alpha) * (self.max - self.min) / delta + self.min
        data.x = data.x[:, torch.nonzero(delta, as_tuple=False).squeeze()]  # remove features with delta = 0
        return data


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
