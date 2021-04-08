import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import accuracy
from torch.nn import Linear, Dropout
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_sparse import matmul
eps = 1e-20


class KProp(MessagePassing):
    def __init__(self, steps, aggregator, add_self_loops, normalize, cached):
        super().__init__(aggr=aggregator)
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self._cached_x = None

    def reset_parameters(self):
        self._cached_x = None

    def forward(self, x, adj_t):
        if self._cached_x is None or not self.cached:
            self._cached_x = self.neighborhood_aggregation(x, adj_t)

        return self._cached_x

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


class KPropConv(KProp):
    def __init__(self, in_channels, out_channels, steps, aggregator, add_self_loops, normalize, cached):
        super().__init__(steps, aggregator, add_self_loops, normalize, cached=cached)
        self.fc = Linear(in_channels, out_channels)

    def forward(self, x, adj_t):
        x = super().forward(x, adj_t)
        x = self.fc(x)
        return x

    def reset_parameters(self):
        super().reset_parameters()
        self.fc.reset_parameters()


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, steps, aggregator, batch_norm, dropout, add_self_loops):
        super().__init__()
        self.conv1 = KPropConv(input_dim, hidden_dim, steps=steps, aggregator=aggregator,
                               add_self_loops=add_self_loops, normalize=True, cached=True)
        self.conv2 = KPropConv(hidden_dim, output_dim, steps=1, aggregator=aggregator,
                               add_self_loops=True, normalize=True, cached=False)
        self.bn = BatchNorm(hidden_dim) if batch_norm else None
        self.dropout = Dropout(p=dropout)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.conv1(x, adj_t)
        x = self.bn(x) if self.bn else x
        x = torch.selu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        x = F.softmax(x, dim=1)
        return x

    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()


class LabelGNN(torch.nn.Module):
    def __init__(self, y_steps):
        super().__init__()
        self.y_steps = y_steps
        self.kprop = KProp(steps=y_steps, aggregator='add', add_self_loops=False, normalize=True, cached=False)

    def forward(self, y, adj_t):
        y = self.kprop(y, adj_t)
        if self.y_steps > 0:
            y = torch.softmax(y, dim=1)
        return y

    def reset_parameters(self):
        self.kprop.reset_parameters()


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim: dict(help='dimension of the hidden layers') = 16,
                 dropout: dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps: dict(help='KProp step parameter', option='-k') = 1,
                 y_steps: dict(help='number of label propagation steps') = 0,
                 propagate_predictions: dict(help='whether to propagate predictions') = False,
                 batch_norm: dict(help='use batch-normalization') = True,
                 add_self_loops: dict(help='whether to add self-loops to the graph') = True,
                 ):
        super().__init__()

        self.propagate_predictions = propagate_predictions
        self.y_steps = y_steps

        self.x_gnn = GNN(
            input_dim=input_dim, output_dim=num_classes, hidden_dim=hidden_dim, steps=x_steps,
            aggregator='add', batch_norm=batch_norm, dropout=dropout, add_self_loops=add_self_loops
        )

        self.y_gnn = LabelGNN(y_steps=y_steps)

    def reset_parameters(self):
        self.x_gnn.reset_parameters()
        self.y_gnn.reset_parameters()

    def forward(self, data):
        return self.x_gnn(data)

    def step(self, data, mask):
        p_y_x = p_yt_x = self.x_gnn(data)  # P(y|x')

        if self.propagate_predictions:
            p_yp_x = torch.matmul(p_y_x, data.T)  # P(y'|x')
            p_yt_x = self.y_gnn(p_yp_x, data.adj_t)  # P(y~|x')

        yp = data.y.float()
        yp[data.test_mask] = 0  # to avoid using test labels
        yt = self.y_gnn(yp, data.adj_t)  # y~

        out = p_yt_x[mask]
        target = yt[mask]
        loss = F.nll_loss(input=torch.log(out + eps), target=target.argmax(dim=1))
        acc = accuracy(pred=out.argmax(dim=1), target=target.argmax(dim=1)) * 100
        return loss, acc

    def training_step(self, data):
        mask = data.train_mask
        loss, acc = self.step(data, mask)
        metrics = {'train_loss': loss.item(), 'train_acc': acc}
        return loss, metrics

    def validation_step(self, data):
        mask = data.val_mask
        loss, acc = self.step(data, mask)
        metrics = {'val_loss': loss.item(), 'val_acc': acc}
        metrics.update(self.test_step(data))
        return metrics

    def test_step(self, data):
        pred = self(data)[data.test_mask].argmax(dim=1)
        target = data.y[data.test_mask].argmax(dim=1)
        acc = accuracy(pred=pred, target=target) * 100
        return {'test_acc': acc}
