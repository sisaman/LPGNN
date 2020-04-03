import math
import torch
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import degree


def get_degree(data):
    row, col = data.edge_index
    return degree(row, data.num_nodes)


def one_bit_response(data, epsilon, priv_dim=-1):
    if priv_dim == -1:
        priv_dim = data.num_node_features
    exp = math.exp(epsilon)
    x = data.x[:, :priv_dim]
    p = (x - data.alpha[:priv_dim]) / data.delta[:priv_dim]
    p[torch.isnan(p)] = 0.  # nan happens when alpha = beta, so also data.x = alpha, so the prev fraction must be 0
    p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
    x = Bernoulli(p).sample()
    data.x = torch.cat([x, data.x[:, priv_dim:]], dim=1)
    return data


@torch.no_grad()
def convert_data(data, feature, **featargs):
    if feature == 'priv':
        return one_bit_response(data, **featargs)
    elif feature == 'locd':
        num_nodes = data.num_nodes
        data.x = None
        data.num_nodes = num_nodes
        return LocalDegreeProfile()(data)
    else:
        return data
