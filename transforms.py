import math
import torch
import torch.nn.functional as F
from mechanisms import supported_mechanisms


class Privatize:
    non_private_methods = ['raw', 'rnd', 'ohd', 'one', 'crnd']
    private_methods = list(supported_mechanisms.keys())

    def __init__(self,
                 method:        dict(help='feature perturbation method', choices=non_private_methods + private_methods,
                                     option=('-m', '--method')) = 'raw',
                 epsilon:       dict(help='privacy budget epsilon (ignored for non-DP methods)', type=float,
                                     option=('-e', '--epsilon')) = None,
                 projection_dim:    dict(help='dimension of the random feature projection', type=int) = None,
                 input_range = None
                 ):

        self.method = method
        self.epsilon = epsilon
        self.projection_dim = projection_dim
        self.input_range = input_range

        assert method in self.non_private_methods or (epsilon is not None and epsilon > 0)

    def __call__(self, data):
        # backup original features for later use
        if not hasattr(data, 'x_raw'):
            data.x_raw = data.x

        # restore original features should they have changed
        data.x = data.x_raw

        if self.method == 'rnd':
            data.x = torch.rand_like(data.x)
            return data

        if self.method == 'crnd':
            n = data.x.size(0)
            d = data.x.size(1)
            m = int(max(1, min(d, math.floor(self.epsilon / 2.18))))
            x = torch.rand(n, m, device=data.x.device)
            s = torch.rand_like(data.x).topk(m, dim=1).indices
            data.x = torch.zeros_like(data.x).scatter(1, s, x)
            return data

        if self.method == 'ohd':
            return OneHotDegree(max_degree=data.num_features - 1)(data)

        elif self.method == 'one':
            data.x = torch.ones_like(data.x)
        elif self.method in self.private_methods:
            if self.input_range is None:
                self.input_range = data.x.min().item(), data.x.max().item()
            data.x = supported_mechanisms[self.method](eps=self.epsilon, input_range=self.input_range)(data.x)

        return data

    @classmethod
    def supported_methods(cls):
        return cls.non_private_methods + cls.private_methods


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
