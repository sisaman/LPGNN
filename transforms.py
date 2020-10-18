import torch
from torch_geometric.transforms import OneHotDegree
from privacy import _available_mechanisms


class FeatureTransform:
    non_private_methods = ['raw', 'rnd', 'ohd']
    private_methods = list(_available_mechanisms.keys())

    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, data):
        if self.method == 'raw':
            return self.restore_features(data)

        # backup original features for later use
        data = self.backup_features(data)

        if self.method == 'rnd':
            data.x = torch.rand_like(data.x_raw)
        elif self.method == 'ohd':
            data = OneHotDegree(max_degree=data.num_features, cat=False)(data)
        elif self.method in self.private_methods:
            data.x = _available_mechanisms[self.method](**self.kwargs)(data.x_raw)

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
    def available_methods(cls):
        return cls.non_private_methods + cls.private_methods


class NodeSplit:
    def __init__(self, val_ratio=.25, test_ratio=.25, random_state=None):
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

        n_val = int(self.val_ratio * num_nodes_with_class)
        n_test = int(self.test_ratio * num_nodes_with_class)
        perm = torch.randperm(num_nodes_with_class, generator=self.rng)

        val_nodes = perm[:n_val]
        test_nodes = perm[n_val:n_val + n_test]
        train_nodes = perm[n_val + n_test:]

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


class LabelRate:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, data):
        if self.rate < 1:
            if not hasattr(data, 'train_mask_full'):
                data.train_mask_full = data.train_mask

            train_idx = data.train_mask_full.nonzero(as_tuple=False).squeeze()
            num_train_nodes = train_idx.size(0)
            train_idx_shuffled = train_idx[torch.randperm(num_train_nodes)]
            train_idx_selected = train_idx_shuffled[:int(self.rate * num_train_nodes)]
            train_mask = torch.zeros_like(data.train_mask_full).scatter(0, train_idx_selected, True)
            data.train_mask = train_mask
        else:
            if hasattr(data, 'train_mask_full'):
                data.train_mask = data.train_mask_full
                del data.train_mask_full
        return data
