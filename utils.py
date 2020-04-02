import math
import torch
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import degree


def get_degree(data):
    row, col = data.edge_index
    return degree(row, data.num_nodes)


def one_bit_response(data, epsilon):
    exp = math.exp(epsilon)
    p = (data.x - data.alpha) / data.delta
    p[torch.isnan(p)] = 0.  # nan happens when alpha = beta, so also data.x = alpha, so the prev fraction must be 0
    p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
    data.x = Bernoulli(p).sample()
    return data


@torch.no_grad()
def convert_data(data, feature, **featargs):
    if feature == 'priv':
        return one_bit_response(data, featargs['epsilon'])
    elif feature == 'locd':
        num_nodes = data.num_nodes
        data.x = None
        data.num_nodes = num_nodes
        return LocalDegreeProfile()(data)
    else:
        return data
