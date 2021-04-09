import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import accuracy
from torch.nn import Dropout
from torch_geometric.nn import MessagePassing, SAGEConv, GCNConv
from torch_sparse import matmul
eps = 1e-20


class KProp(MessagePassing):
    def __init__(self, steps, aggregator, add_self_loops, normalize, cached, post_step = lambda x: x):
        super().__init__(aggr=aggregator)
        self.post_step = post_step
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
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
            x = self.post_step(x)

        return x

    def message_and_aggregate(self, adj_t, x):  # noqa
        return matmul(adj_t, x, reduce=self.aggr)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = Dropout(p=dropout)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = torch.selu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, normalize=False, root_weight=True)
        self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=output_dim, normalize=False, root_weight=True)
        self.dropout = Dropout(p=dropout)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = torch.selu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        return x


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 model:                 dict(help='backbone GNN model', choices=['gcn', 'sage']) = 'sage',
                 hidden_dim:            dict(help='dimension of the hidden layers') = 16,
                 dropout:               dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps:               dict(help='KProp step parameter', option='-k') = 1,
                 y_steps:               dict(help='number of label propagation steps') = 0,
                 propagate_predictions: dict(help='whether to propagate predictions') = False,
                 ):
        super().__init__()

        self.propagate_predictions = propagate_predictions
        self.y_steps = y_steps

        self.x_prop = KProp(steps=x_steps, aggregator='add', add_self_loops=False, normalize=True, cached=True)
        self.y_prop = KProp(steps=y_steps, aggregator='add', add_self_loops=False, normalize=True, cached=False)

        if model == 'sage':
            self.gnn = GraphSAGE(input_dim=input_dim, output_dim=num_classes, hidden_dim=hidden_dim, dropout=dropout)
        elif model == 'gcn':
            self.gnn = GCN(input_dim=input_dim, output_dim=num_classes, hidden_dim=hidden_dim, dropout=dropout)

        self.cached_yt = None

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.x_prop(x, adj_t)
        x = self.gnn(x, adj_t)
        x = F.softmax(x, dim=1)
        return x

    def step(self, data, mask):
        p_y_x = p_yt_x = self(data)  # P(y|x')

        if self.propagate_predictions:
            p_yp_x = torch.matmul(p_y_x, data.T)  # P(y'|x')
            p_yt_x = self.y_prop(p_yp_x, data.adj_t)  # P(y~|x')

        if self.cached_yt is None:
            yp = data.y.float()
            yp[data.test_mask] = 0  # to avoid using test labels
            self.cached_yt = self.y_prop(yp, data.adj_t)  # y~

        out = p_yt_x[mask]
        target = self.cached_yt[mask]
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
