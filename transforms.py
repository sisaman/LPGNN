import torch
import torch.nn.functional as F
from mechanisms import supported_feature_mechanisms, RandomizedResopnse
from torch_sparse import matmul

from utils import one_hot_encode


class FeatureTransform:
    supported_features = ['raw', 'rnd', 'one', 'ohd', 'crnd']

    def __init__(self, feature: dict(help='feature transformation method',
                                     choices=supported_features, option='-f') = 'raw',
                 ):

        self.feature = feature

    def __call__(self, data):

        if self.feature == 'rnd':
            data.x = torch.rand_like(data.x)
        elif self.feature == 'crnd':
            n = data.x.size(0)
            # d = data.x.size(1)
            m = 1
            x = torch.rand(n, m, device=data.x.device)
            s = torch.rand_like(data.x).topk(m, dim=1).indices
            data.x = torch.zeros_like(data.x).scatter(1, s, x)
        elif self.feature == 'ohd':
            data = OneHotDegree(max_degree=data.num_features - 1)(data)
        elif self.feature == 'one':
            data.x = torch.zeros_like(data.x)

        return data


class FeaturePerturbation:
    def __init__(self,
                 mechanism: dict(help='feature perturbation mechanism', choices=list(supported_feature_mechanisms),
                                 option='-m') = 'mbm',
                 x_eps: dict(help='privacy budget for feature perturbation (set None to disable)', type=float,
                             option='-ex') = None,
                 reduce_dim: dict(help='dimension of the random dimensionality reduction (set None to disable)',
                                  type=int) = None,
                 data_range=None,
                 ):
        self.mechanism = mechanism
        self.input_range = data_range
        self.reduce_dim = reduce_dim
        self.x_eps = x_eps

    def __call__(self, data):
        if self.x_eps is None:
            return data

        if self.input_range is None:
            self.input_range = data.x.min().item(), data.x.max().item()

        if self.reduce_dim:
            data = RandomizedProjection(input_dim=data.num_features, output_dim=self.reduce_dim)(data)

        data.x = supported_feature_mechanisms[self.mechanism](
            eps=self.x_eps,
            input_range=self.input_range
        )(data.x)

        return data


class LabelPerturbation:
    def __init__(self,
                 y_eps: dict(help='privacy budget for label perturbation (set None to disable)',
                             type=float, option='-ey') = None,
                 ):
        self.y_eps = y_eps

    def __call__(self, data):
        data.y = one_hot_encode(data.y, num_classes=data.num_classes)
        p_ii = 1  # probability of preserving the clean label i
        p_ij = 0  # probability of perturbing label i into another label j

        if self.y_eps is not None:
            mechanism = RandomizedResopnse(eps=self.y_eps, d=data.num_classes)
            perturb_mask = data.train_mask | data.val_mask
            y_perturbed = mechanism(data.y[perturb_mask])
            data.y[perturb_mask] = y_perturbed
            p_ii, p_ij = mechanism.p, mechanism.q

        # set label transistion matrix
        data.T = torch.ones(data.num_classes, data.num_classes, device=data.y.device) * p_ij
        data.T.fill_diagonal_(p_ii)

        return data


class RandomizedProjection:
    def __init__(self, input_dim, output_dim):
        self.w = torch.rand(input_dim, output_dim)

    def __call__(self, data):
        data.x = data.x.matmul(self.w)
        return data


class OneHotDegree:
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, data):
        degree = data.adj_t.sum(dim=0).long()
        degree.clamp_(max=self.max_degree)
        data.x = F.one_hot(degree, num_classes=self.max_degree + 1).float()  # add 1 for zero degree
        return data


class NodeSplit:
    def __init__(self, train_ratio=None, val_ratio=.25, test_ratio=.25, random_state=None):
        self.train_ratio = 1 - (val_ratio + test_ratio) if train_ratio is None else train_ratio
        assert self.train_ratio > 0
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = None
        if random_state is not None:
            self.rng = torch.Generator().manual_seed(random_state)

    def __call__(self, data):
        num_nodes_with_class = data.num_nodes
        nodes_with_class = torch.ones(data.num_nodes, dtype=torch.bool)

        if hasattr(data, 'y') and -1 in data.y:
            nodes_with_class = data.y != -1
            num_nodes_with_class = nodes_with_class.sum().item()

        n_train = int(self.train_ratio * num_nodes_with_class)
        n_val = int(self.val_ratio * num_nodes_with_class)
        n_test = int(self.test_ratio * num_nodes_with_class)
        perm = torch.randperm(num_nodes_with_class, generator=self.rng)

        train_nodes = perm[:n_train]
        val_nodes = perm[n_train: n_train + n_val]
        test_nodes = perm[n_train + n_val: n_train + n_val + n_test]

        temp_val_mask = torch.zeros(num_nodes_with_class, dtype=torch.bool)
        temp_val_mask[val_nodes] = True

        temp_test_mask = torch.zeros(num_nodes_with_class, dtype=torch.bool)
        temp_test_mask[test_nodes] = True

        temp_train_mask = torch.zeros(num_nodes_with_class, dtype=torch.bool)
        temp_train_mask[train_nodes] = True

        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        val_mask[nodes_with_class] = temp_val_mask
        test_mask[nodes_with_class] = temp_test_mask
        train_mask[nodes_with_class] = temp_train_mask

        data.val_mask = val_mask
        data.test_mask = test_mask
        data.train_mask = train_mask
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
