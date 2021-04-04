import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import accuracy
from torch.nn import Linear, Dropout
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_sparse import matmul


class KProp(MessagePassing):
    def __init__(self, in_channels, out_channels, steps, aggregator, add_self_loops, normalize, cached=False):
        super().__init__(aggr=aggregator)
        self.fc = Linear(in_channels, out_channels)
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self._cached_x = None

    def forward(self, x, adj_t):
        if self._cached_x is None or not self.cached:
            x = self.neighborhood_aggregation(x, adj_t)
            self._cached_x = x

        x = self.fc(self._cached_x)
        return x

    def neighborhood_aggregation(self, x, adj_t):
        if self.K <= 0:
            return x

        if self.normalize:
            adj_t = gcn_norm(adj_t, add_self_loops=False)

        if self.add_self_loops:
            adj_t = adj_t.set_diag()

        for k in range(self.K):
            x = self.propagate(adj_t, x=x)

        return x

    def message_and_aggregate(self, adj_t, x):  # noqa
        return matmul(adj_t, x, reduce=self.aggr)


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, steps, aggregator, batch_norm, dropout, add_self_loops):
        super().__init__()
        self.conv1 = KProp(input_dim, hidden_dim, steps=steps, aggregator=aggregator,
                           add_self_loops=add_self_loops, normalize=True, cached=True)
        self.conv2 = KProp(hidden_dim, output_dim, steps=1, aggregator=aggregator,
                           add_self_loops=True, normalize=True, cached=False)
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


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim: dict(help='dimension of the hidden layers') = 16,
                 dropout: dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps: dict(help='KProp step parameter', option='-k') = 1,
                 y_steps: dict(help='number of label propagation steps') = 0,
                 batch_norm: dict(help='use batch-normalization') = True,
                 add_self_loops: dict(help='whether to add self-loops to the graph') = True,
                 ):
        super().__init__()

        self.y_steps = y_steps

        self.gnn = GNN(
            input_dim=input_dim, output_dim=num_classes, hidden_dim=hidden_dim, steps=x_steps,
            aggregator='add', batch_norm=batch_norm, dropout=dropout, add_self_loops=add_self_loops
        )
        self.prop = KProp(
            1, 1, steps=y_steps, aggregator='add', add_self_loops=False, normalize=True, cached=False
        ).neighborhood_aggregation

    def forward(self, data):
        return self.gnn(data)

    def training_step(self, data):
        mask = data.train_mask
        p_y_x = self(data)
        p_yp_x = torch.matmul(p_y_x, data.T)
        yt_x = self.prop(p_yp_x, data.adj_t)

        yt_yp = data.y.float()
        yt_yp[data.test_mask] = 0  # to avoid using test labels
        p_yt_yp = self.prop(yt_yp, data.adj_t)

        if self.y_steps > 0:
            log_p_yt_x = torch.log_softmax(yt_x, dim=1)
            p_yt_yp = torch.softmax(p_yt_yp, dim=1)
        else:
            log_p_yt_x = torch.log(yt_x + 1e-7)

        out = log_p_yt_x[mask]
        target = p_yt_yp[mask].argmax(dim=1)

        loss = F.nll_loss(input=out, target=target)
        acc = accuracy(pred=out.argmax(dim=1), target=target) * 100
        metrics = {'train_loss': loss.item(), 'train_acc': acc}
        return loss, metrics

    def validation_step(self, data):
        mask = data.val_mask
        p_y_x = self(data)
        p_yp_x = torch.matmul(p_y_x, data.T)
        out = torch.log(p_yp_x[mask] + 1e-8)
        target = data.y[mask].argmax(dim=1)

        loss = F.nll_loss(input=out, target=target)
        acc = accuracy(pred=out.argmax(dim=1), target=target) * 100
        metrics = {'val_loss': loss.item(), 'val_acc': acc}
        metrics.update(self.test_step(data))
        return metrics

    def test_step(self, data):
        pred = self(data)[data.test_mask].argmax(dim=1)
        target = data.y[data.test_mask].argmax(dim=1)
        acc = accuracy(pred=pred, target=target) * 100
        return {'test_acc': acc}
