import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.5, inductive=False):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=not inductive)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=not inductive)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, inductive=False):
        super().__init__()
        self.conv = GCNConv(input_dim, 2 * output_dim, cached=not inductive)
        self.bn = BatchNorm1d(2 * output_dim)
        self.conv_mu = GCNConv(2 * output_dim, output_dim, cached=not inductive)
        self.conv_logvar = GCNConv(2 * output_dim, output_dim, cached=not inductive)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
