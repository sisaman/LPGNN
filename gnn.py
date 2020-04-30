import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree


class GConv(MessagePassing):
    def __init__(self, cached=False):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.cached = cached
        self.cached_gc = None

    def forward(self, x, edge_index):
        if not self.cached or self.cached_gc is None:
            num_nodes = x.size(0)
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
            row, col = edge_index
            deg = degree(row, num_nodes, dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).view(-1, 1)
            self.cached_gc = self.propagate(edge_index, x=x, norm=norm)
        return self.cached_gc

    # noinspection PyMethodOverriding
    def message(self, x_j, norm):
        return norm * x_j


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.5, cached=True):
        super().__init__()
        self.conv1 = GConv(cached=cached)
        self.lin1 = Linear(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=cached)
        self.dropout = dropout
        self.cached = cached
        self.cached_gc = None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.lin1(x)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, cached=True):
        super().__init__()
        self.conv1 = GConv(cached=cached)
        self.lin1 = Linear(input_dim, 2 * output_dim)
        self.bn = BatchNorm1d(2 * output_dim)
        self.conv_mu = GCNConv(2 * output_dim, output_dim, cached=cached)
        self.conv_logvar = GCNConv(2 * output_dim, output_dim, cached=cached)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.lin1(x)
        x = self.bn(x)
        x = F.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
