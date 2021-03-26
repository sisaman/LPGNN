import torch
import torch.nn.functional as F
from torch_geometric.utils import accuracy
from torch.nn import Linear, Dropout
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul


class KProp(MessagePassing):
    def __init__(self, in_channels, out_channels, step, aggregator, add_self_loops, cached=False):
        super().__init__(aggr='add' if aggregator == 'gcn' else aggregator)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = Linear(in_channels, out_channels)
        self.K = step
        self.add_self_loops = add_self_loops
        self.cached = cached
        self._cached_x = None
        self.aggregator = aggregator

    def forward(self, x, adj_t):
        if self._cached_x is None or not self.cached:
            x = self.neighborhood_aggregation(x, adj_t)
            self._cached_x = x

        x = self.fc(self._cached_x)
        return x

    def neighborhood_aggregation(self, x, adj_t):
        if self.aggregator == 'gcn':
            adj_t = gcn_norm(
                adj_t, num_nodes=x.size(self.node_dim),
                add_self_loops=self.add_self_loops, dtype=x.dtype
            )
        elif self.add_self_loops:
            adj_t = adj_t.set_diag()

        for k in range(self.K):
            x = self.propagate(adj_t, x=x)

        return x

    # noinspection PyMethodOverriding
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim: dict(help='dimension of the hidden layers') = 16,
                 dropout:    dict(help='dropout rate (between zero and one)') = 0.0,
                 step:       dict(help='KProp step parameter', option='-k') = 1,
                 aggregator: dict(help='GNN aggregator function', choices=['gcn', 'mean']) = 'gcn',
                 batch_norm: dict(help='use batch-normalization') = True,
                 self_loops: dict(help='whether to add self-loops to the graph') = True,
                 ):
        super().__init__()

        self.conv1 = KProp(input_dim, hidden_dim, step=step, aggregator=aggregator,
                           add_self_loops=self_loops, cached=True)
        self.conv2 = KProp(hidden_dim, num_classes, step=1, aggregator=aggregator,
                           add_self_loops=True, cached=False)
        self.bn = BatchNorm(hidden_dim) if batch_norm else None
        self.dropout = Dropout(p=dropout)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.conv1(x, adj_t)
        if self.bn:
            x = self.bn(x)
        x = torch.selu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        x = F.log_softmax(x, dim=1)
        return x

    def evaluate(self, data, mask):
        out = self(data)
        loss = F.nll_loss(out[mask], data.y[mask])
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[mask], target=data.y[mask]) * 100
        return loss, acc

    def training_step(self, data):
        loss, acc = self.evaluate(data, mask=data.train_mask)
        return {'train_loss': loss, 'train_acc': acc}

    def validation_step(self, data):
        loss, acc = self.evaluate(data, mask=data.val_mask)
        return {'val_loss': loss.item(), 'val_acc': acc}

    def test_step(self, data):
        out = self(data)
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[data.test_mask], target=data.y[data.test_mask]) * 100
        return {'test_acc': acc}
