import torch
import torch.nn.functional as F
from torch_geometric.utils import accuracy
from torch.nn import Linear, Dropout
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_sparse import matmul


class KProp(MessagePassing):
    def __init__(self, in_channels, out_channels, steps, aggregator, add_self_loops, cached=False):
        super().__init__(aggr=aggregator)
        self.fc = Linear(in_channels, out_channels)
        self.K = steps
        self.add_self_loops = add_self_loops
        self.cached = cached
        self._cached_x = None

    def forward(self, x, adj_t):
        if self._cached_x is None or not self.cached:
            x = self.neighborhood_aggregation(x, adj_t)
            self._cached_x = x

        x = self.fc(self._cached_x)
        return x

    def neighborhood_aggregation(self, x, adj_t):
        if self.add_self_loops:
            adj_t = adj_t.set_diag()

        for k in range(self.K):
            x = self.propagate(adj_t, x=x)

        return x

    # noinspection PyMethodOverriding
    def message_and_aggregate(self, adj_t, x):
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
        x = F.log_softmax(x, dim=1)
        return x


class LabelGNN(torch.nn.Module):  # todo add second layer
    def __init__(self, num_classes, steps, aggregator):
        super().__init__()
        self.conv = KProp(
            in_channels=num_classes, out_channels=num_classes, steps=steps,
            aggregator=aggregator, add_self_loops=False, cached=True
        )

    def forward(self, y, adj_t):
        y = self.conv(y, adj_t)
        y = torch.log_softmax(y, dim=1)
        return y


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim:        dict(help='dimension of the hidden layers') = 16,
                 dropout:           dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps:           dict(help='KProp step parameter', option='-k') = 1,
                 y_steps: dict(help='number of label propagation steps') = 0,
                 aggregator:        dict(help='GNN aggregator function', choices=['add', 'mean']) = 'add',
                 batch_norm:        dict(help='use batch-normalization') = True,
                 add_self_loops:    dict(help='whether to add self-loops to the graph') = True,
                 ):
        super().__init__()

        self.y_steps = y_steps
        self.theta = torch.nn.Parameter(torch.Tensor(self.y_steps + 1))
        self.x_gnn = GNN(
            input_dim=input_dim, output_dim=num_classes, hidden_dim=hidden_dim, steps=x_steps,
            aggregator=aggregator, batch_norm=batch_norm, dropout=dropout, add_self_loops=add_self_loops
        )
        self.y_gnn = LabelGNN(num_classes=num_classes, steps=y_steps, aggregator=aggregator)

    def forward(self, data):
        return self.x_gnn(data)

    def infer_labels(self, data):
        mask = data.train_mask | data.val_mask
        adj_t = data.adj_t[mask, mask]
        s = data.y[mask]
        theta = torch.softmax(self.theta, dim=0)
        y = theta[0] * s

        for i in range(1, self.y_steps + 1):
            s = matmul(adj_t, s, reduce='sum')
            y += theta[i] * s

        new_y = data.y.clone()
        new_y[mask] = y
        new_y = new_y.argmax(dim=1)
        return new_y

    def training_step(self, data):
        mask = data.train_mask
        out = self(data)
        target = self.infer_labels(data)
        loss = F.nll_loss(out[mask], target[mask])
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[mask], target=target[mask]) * 100
        # loss, acc = self.evaluate(data, mask=data.train_mask)
        return {'train_loss': loss, 'train_acc': acc}

    def validation_step(self, data):
        mask = data.val_mask
        out = self(data)
        target = self.infer_labels(data)
        loss = F.nll_loss(out[mask], target[mask])
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[mask], target=target[mask]) * 100
        # loss, acc = self.evaluate(data, mask=data.val_mask)
        # test_loss, test_acc = self.evaluate(data, mask=data.test_mask)
        result = {'val_loss': loss.item(), 'val_acc': acc}
        # result.update({'test_loss': test_loss.item(), 'test_acc': test_acc})
        return result

    def test_step(self, data):
        # print(self.theta)
        out = self(data)
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[data.test_mask], target=data.y[data.test_mask].argmax(dim=1)) * 100
        return {'test_acc': acc}
