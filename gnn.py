import math
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree


# noinspection PyMethodOverriding
class GConvDP(MessagePassing):
    @staticmethod
    def norm(edge_index, num_nodes, dtype):
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        return edge_index, (deg_inv_sqrt[row] * deg_inv_sqrt[col]).view(-1, 1)

    def __init__(self, epsilon=1, alpha=0, delta=0, cached=False):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.eps = epsilon
        self.alpha = alpha
        self.delta = delta
        self.cached = cached
        self.cached_gc = None

    def forward(self, x, edge_index, priv_mask):
        if not self.cached or self.cached_gc is None:
            num_nodes = x.size(0)
            edge_index, norm = self.norm(edge_index, num_nodes, x.dtype)
            self.cached_gc = self.propagate(edge_index, x=x, norm=norm, p=priv_mask)
        return self.cached_gc

    def message(self, x_j, p_j, norm):
        exp = math.exp(self.eps)
        msg = norm * (p_j * ((((exp + 1) * x_j - 1) * self.delta) / (exp - 1) + self.alpha) + (~p_j) * x_j)
        return msg

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.5, epsilon=1, alpha=0, delta=0, cached=True):
        assert epsilon > 0
        super().__init__()
        self.conv1 = GConvDP(epsilon, alpha, delta, cached=cached)
        self.lin1 = Linear(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=cached)
        self.dropout = dropout
        self.cached = cached
        self.cached_gc = None

    def forward(self, x, edge_index, priv_mask):
        x = self.conv1(x, edge_index, priv_mask)
        x = self.lin1(x)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, epsilon=1, alpha=0, delta=0, cached=True):
        super().__init__()
        self.conv1 = GConvDP(epsilon, alpha, delta, cached=cached)
        self.lin1 = Linear(input_dim, 2 * output_dim)
        self.conv_mu = GCNConv(2 * output_dim, output_dim, cached=cached)
        self.conv_logvar = GCNConv(2 * output_dim, output_dim, cached=cached)

    def forward(self, x, edge_index, priv_mask):
        x = self.conv1(x, edge_index, priv_mask)
        x = self.lin1(x)
        x = torch.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
