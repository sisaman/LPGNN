import torch
import torch.nn.functional as F
from mechanisms import supported_mechanisms


class Privatize:
    non_private_methods = ['raw', 'rnd', 'ohd']
    private_methods = list(supported_mechanisms.keys())

    def __init__(self,
                 method: dict(help='feature perturbation method', choices=non_private_methods + private_methods,
                              option=('-m', '--method')) = 'raw',
                 epsilon: dict(help='privacy budget epsilon (ignored for non-DP methods)', type=float,
                               option=('-e', '--epsilon')) = None,
                 **kwargs):
        self.method = method
        self.epsilon = epsilon
        self.kwargs = kwargs

        assert method in self.non_private_methods or (epsilon is not None and epsilon > 0)

    def __call__(self, data):
        if self.method == 'raw':
            return self.restore_features(data)

        # backup original features for later use
        data = self.backup_features(data)

        if self.method == 'rnd':
            data.x = torch.rand_like(data.x_raw)
        elif self.method == 'ohd':
            data = OneHotDegree(max_degree=data.num_features - 1)(data)
        elif self.method in self.private_methods:
            data.x = supported_mechanisms[self.method](eps=self.epsilon, **self.kwargs)(data.x_raw)

        return data

    @staticmethod
    def backup_features(data):
        if not hasattr(data, 'x_raw'):
            data.x_raw = data.x
        return data

    @staticmethod
    def restore_features(data):
        if hasattr(data, 'x_raw'):
            del data.x              # delete x to free up memory
            data.x = data.x_raw     # bring back original features
        return data

    @classmethod
    def supported_methods(cls):
        return cls.non_private_methods + cls.private_methods


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
