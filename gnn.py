import math

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation.

    @staticmethod
    def norm(edge_index, num_nodes, dtype):
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        return edge_index, (deg_inv_sqrt[row] * deg_inv_sqrt[col]).view(-1, 1)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        edge_index, norm = self.norm(edge_index, num_nodes, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)

        return x

    # noinspection PyMethodOverriding
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        return norm * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


# noinspection PyMethodOverriding
class GConvDP(GCNConv):
    def __init__(self, epsilon, alpha, delta):
        super().__init__()
        self.eps = epsilon
        self.alpha = alpha
        self.delta = delta

    def message(self, x_j, norm):
        exp = math.exp(self.eps)
        return norm * ((((exp + 1) * x_j - 1) * self.delta) / (exp - 1) + self.alpha)


class GConvMixedDP(torch.nn.Module):
    def __init__(self, priv_dim, epsilon, alpha, delta):
        super().__init__()
        self.priv_dim = priv_dim
        self.gcnconv = GCNConv()
        self.gconvdp = GConvDP(epsilon, alpha[:priv_dim], delta[:priv_dim])

    def forward(self, x, edge_index):
        x_left = self.gconvdp(x[:, :self.priv_dim], edge_index)
        x_right = self.gcnconv(x[:, self.priv_dim:], edge_index)
        x = torch.cat([x_left, x_right], dim=1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, priv_input_dim=0, **dpargs):
        super().__init__()

        if priv_input_dim == 0:
            self.conv1 = GCNConv()
        elif priv_input_dim == input_dim:
            self.conv1 = GConvDP(**dpargs)
        else:
            self.conv1 = GConvMixedDP(priv_input_dim, **dpargs)

        self.lin1 = Linear(input_dim, hidden_dim)
        self.conv2 = GCNConv()
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.lin2(x)
        return x
