import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import accuracy as accuracy_1d
from torch.nn import Dropout, SELU
from torch_geometric.nn import MessagePassing, SAGEConv, GCNConv, GATConv
from torch_sparse import matmul


class KProp(MessagePassing):
    def __init__(self, steps, aggregator, add_self_loops, normalize, cached, transform=lambda x: x):
        super().__init__(aggr=aggregator)
        self.transform = transform
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

        x = self.transform(x)
        return x

    def message_and_aggregate(self, adj_t, x):  # noqa
        return matmul(adj_t, x, reduce=self.aggr)


class GNN(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv1 = None
        self.conv2 = None
        self.dropout = Dropout(p=dropout)
        self.activation = SELU(inplace=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        return x


class GCN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)


class GAT(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        heads = 4
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(heads * hidden_dim, output_dim, heads=1, concat=False)


class GraphSAGE(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, normalize=False, root_weight=True)
        self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=output_dim, normalize=False, root_weight=True)


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 model:                 dict(help='backbone GNN model', choices=['gcn', 'sage', 'gat']) = 'sage',
                 hidden_dim:            dict(help='dimension of the hidden layers') = 16,
                 dropout:               dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps:               dict(help='KProp step parameter for features', option='-kx') = 0,
                 y_steps:               dict(help='KProp step parameter for labels', option='-ky') = 0,
                 forward_correction:    dict(help='applies forward loss correction') = True,
                 ):
        super().__init__()

        self.x_prop = KProp(steps=x_steps, aggregator='add', add_self_loops=False, normalize=True, cached=True)
        self.y_prop = KProp(steps=y_steps, aggregator='add', add_self_loops=False, normalize=True, cached=False,
                            transform=torch.nn.Softmax(dim=1))

        self.gnn = {'gcn': GCN, 'sage': GraphSAGE, 'gat': GAT}[model](
            input_dim=input_dim,
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.cached_yt = None
        self.forward_correction = forward_correction

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.x_prop(x, adj_t)
        x = self.gnn(x, adj_t)

        p_y_x = F.softmax(x, dim=1)                                                         # P(y|x')
        p_yp_x = torch.matmul(p_y_x, data.T) if self.forward_correction else p_y_x          # P(y'|x')
        p_yt_x = self.y_prop(p_yp_x, data.adj_t)                                            # P(y~|x')

        return p_y_x, p_yp_x, p_yt_x

    def training_step(self, data):
        p_y_x, p_yp_x, p_yt_x = self(data)

        if self.cached_yt is None:
            yp = data.y.float()
            yp[data.test_mask] = 0  # to avoid using test labels
            self.cached_yt = self.y_prop(yp, data.adj_t)  # y~

        loss = self.cross_entropy_loss(p_y=p_yt_x[data.train_mask], y=self.cached_yt[data.train_mask], weighted=False)

        metrics = {
            'train/loss': loss.item(),
            'train/acc': self.accuracy(pred=p_y_x[data.train_mask], target=data.y[data.train_mask]) * 100,
            'train/maxacc': data.T[0, 0].item() * 100,
        }

        return loss, metrics

    def validation_step(self, data):
        p_y_x, p_yp_x, p_yt_x = self(data)

        metrics = {
            'val/loss': self.cross_entropy_loss(p_yp_x[data.val_mask], data.y[data.val_mask]).item(),
            'val/acc': self.accuracy(pred=p_y_x[data.val_mask], target=data.y[data.val_mask]) * 100,
            'test/acc': self.accuracy(pred=p_y_x[data.test_mask], target=data.y[data.test_mask]) * 100,
        }

        return metrics

    @staticmethod
    def accuracy(pred, target):
        pred = pred.argmax(dim=1) if len(pred.size()) > 1 else pred
        target = target.argmax(dim=1) if len(target.size()) > 1 else target
        return accuracy_1d(pred=pred, target=target)

    @staticmethod
    def cross_entropy_loss(p_y, y, weighted=False):
        y_onehot = F.one_hot(y.argmax(dim=1))
        loss = -torch.log(p_y + 1e-20) * y_onehot
        loss *= y if weighted else 1
        loss = loss.sum(dim=1).mean()
        return loss