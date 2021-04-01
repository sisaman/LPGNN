import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import accuracy, add_remaining_self_loops
from torch.nn import Linear, Dropout
from torch_geometric.nn import MessagePassing, BatchNorm, APPNP
from torch_sparse import matmul, SparseTensor


class KProp(MessagePassing):
    def __init__(self, in_channels, out_channels, steps, aggregator, add_self_loops, normalize=False, cached=False):
        super().__init__(aggr=aggregator)
        self.fc = Linear(in_channels, out_channels)
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self._cached_x = None

    def forward(self, x, edge_index_or_adj_t, edge_weight=None):
        if self._cached_x is None or not self.cached:
            x = self.neighborhood_aggregation(x, edge_index_or_adj_t, edge_weight)
            self._cached_x = x

        x = self.fc(self._cached_x)
        return x

    def neighborhood_aggregation(self, x, edge_index_or_adj_t, edge_weight=None):
        if self.normalize:
            if isinstance(edge_index_or_adj_t, SparseTensor):
                edge_index_or_adj_t = gcn_norm(edge_index_or_adj_t, add_self_loops=False)
            else:
                edge_index_or_adj_t, edge_weight = gcn_norm(
                    edge_index_or_adj_t, edge_weight, num_nodes=x.size(0), add_self_loops=False
                )

        if self.add_self_loops:
            if isinstance(edge_index_or_adj_t, SparseTensor):
                edge_index_or_adj_t = edge_index_or_adj_t.set_diag()
            else:
                edge_index_or_adj_t, edge_weight = add_remaining_self_loops(
                    edge_index_or_adj_t, edge_weight, num_nodes=x.size(0)
                )

        for k in range(self.K):
            x = self.propagate(edge_index_or_adj_t, x=x)

        return x

    def message(self, x_j, edge_weight):  # noqa
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):  # noqa
        return matmul(adj_t, x, reduce=self.aggr)


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, steps, aggregator, batch_norm, dropout, add_self_loops):
        super().__init__()
        self.conv1 = KProp(input_dim, hidden_dim, steps=steps, aggregator=aggregator,
                           add_self_loops=add_self_loops, cached=True)
        self.conv2 = KProp(hidden_dim, output_dim, steps=1, aggregator=aggregator,
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
        x = F.softmax(x, dim=1)
        return x


class LabelGNN(torch.nn.Module):
    def __init__(self, num_classes, steps, aggregator, dropout):
        super().__init__()
        self.conv1 = KProp(
            in_channels=num_classes, out_channels=num_classes, steps=steps,
            aggregator=aggregator, add_self_loops=False, cached=False
        )

    def forward(self, y, adj_t):
        y = self.conv1.neighborhood_aggregation(y, adj_t)
        y = F.log_softmax(y, dim=1)
        return y


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim: dict(help='dimension of the hidden layers') = 16,
                 dropout: dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps: dict(help='KProp step parameter', option='-k') = 1,
                 y_steps: dict(help='number of label propagation steps') = 0,
                 aggregator: dict(help='GNN aggregator function', choices=['add', 'mean']) = 'add',
                 batch_norm: dict(help='use batch-normalization') = True,
                 add_self_loops: dict(help='whether to add self-loops to the graph') = True,
                 ):
        super().__init__()

        self.y_steps = y_steps

        self.x_gnn = GNN(
            input_dim=input_dim, output_dim=num_classes, hidden_dim=hidden_dim, steps=x_steps,
            aggregator=aggregator, batch_norm=batch_norm, dropout=dropout, add_self_loops=add_self_loops
        )
        self.y_gnn = LabelGNN(num_classes=num_classes, steps=y_steps, aggregator=aggregator, dropout=dropout)

    def forward(self, data):
        return self.x_gnn(data)

    def evaluate(self, data, mask):

        if self.y_steps == 0:
            out = self(data)[mask]
            target = data.y[mask]
        else:
            p_y_x = self(data)
            p_yp_x = torch.matmul(p_y_x, data.p)
            p_yt_x = self.y_gnn(p_yp_x, data.adj_t)[mask]

            yp = data.y.clone()
            yp_test = torch.randint(low=0, high=data.num_classes, size=(data.test_mask.sum(), 1), device=data.y.device)
            yp[data.test_mask] = 0
            yp[data.test_mask].scatter_(1, yp_test, 1)
            p_yt_yp = self.y_gnn(yp, data.adj_t)[mask]

            out = p_yt_x
            target = p_yt_yp

        target = target.argmax(dim=1)
        loss = F.nll_loss(input=out, target=target)
        acc = accuracy(pred=out.argmax(dim=1), target=target) * 100
        return loss, acc

    def training_step(self, data):
        mask = data.train_mask
        loss, acc = self.evaluate(data, mask=mask)
        metrics = {'train_loss': loss.item(), 'train_acc': acc}
        return loss, metrics

    def validation_step(self, data):
        mask = data.val_mask
        loss, acc = self.evaluate(data, mask=mask)
        metrics = {'val_loss': loss.item(), 'val_acc': acc}
        metrics.update(self.test_step(data))
        return metrics

    def test_step(self, data):
        out = self(data)[data.test_mask]
        pred = out.argmax(dim=1)
        target = data.y[data.test_mask].argmax(dim=1)
        acc = accuracy(pred=pred, target=target) * 100
        return {'test_acc': acc}


