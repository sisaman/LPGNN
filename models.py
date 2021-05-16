import torch
import torch.nn.functional as F
from torch.nn import Dropout, SELU
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv


class KProp(torch.nn.Module):
    def __init__(self, steps, aggregator, add_self_loops, normalize, cached, transform=lambda x: x):
        super().__init__()
        self.aggregator = aggregator
        self.transform = transform
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self._cached_x = None

    def forward(self, g, x):
        if self._cached_x is None or not self.cached:
            self._cached_x = self.neighborhood_aggregation(g, x)

        return self._cached_x

    def neighborhood_aggregation(self, g, x):
        if self.K <= 0:
            return x

        conv = GraphConv(
            in_feats=x.size(1),
            out_feats=x.size(1),
            norm='both' if self.normalize else 'none',
            weight=False,
            bias=False
        )

        if self.add_self_loops:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)

        for k in range(self.K):
            x = conv(g, x)

        x = self.transform(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv1 = None
        self.conv2 = None
        self.dropout = Dropout(p=dropout)
        self.activation = SELU(inplace=True)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        return x


class GCN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)


class GAT(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        heads = 4
        self.conv1 = GATConv(input_dim, hidden_dim, num_heads=heads)
        self.conv2 = GATConv(heads * hidden_dim, output_dim, num_heads=1)

    def forward(self, g, x):
        x = self.conv1(g, x).flatten(1)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(g, x).mean(1)
        return x


class GraphSAGE(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_dim, output_dim, aggregator_type='mean')


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 model:                 dict(help='backbone GNN model', choices=['gcn', 'sage', 'gat']) = 'sage',
                 hidden_dim:            dict(help='dimension of the hidden layers') = 16,
                 dropout:               dict(help='dropout rate (between zero and one)') = 0.0,
                 x_steps:               dict(help='KProp step parameter for features', option='-kx') = 0,
                 y_steps:               dict(help='KProp step parameter for labels', option='-ky') = 0,
                 forward_correction:    dict(help='applies forward loss correction', option='--forward') = True,
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

    def forward(self, g):
        x = g.ndata['feat']
        x = self.x_prop(g, x)
        x = self.gnn(g, x)

        p_y_x = F.softmax(x, dim=1)                                                      # P(y|x')
        p_yp_x = torch.matmul(p_y_x, g.T) if self.forward_correction else p_y_x          # P(y'|x')
        p_yt_x = self.y_prop(g, p_yp_x)                                            # P(y~|x')

        return p_y_x, p_yp_x, p_yt_x

    def training_step(self, g):
        p_y_x, p_yp_x, p_yt_x = self(g)

        if self.cached_yt is None:
            yp = g.ndata['label'].float()
            yp[g.ndata['test_mask']] = 0  # to avoid using test labels
            self.cached_yt = self.y_prop(g, yp)  # y~

        train_mask = g.ndata['train_mask']
        loss = self.cross_entropy_loss(p_y=p_yt_x[train_mask], y=self.cached_yt[train_mask], weighted=False)

        metrics = {
            'train/loss': loss.item(),
            'train/acc': self.accuracy(pred=p_y_x[train_mask], target=g.ndata['label'][train_mask]) * 100,
            'train/maxacc': g.T[0, 0].item() * 100,
        }

        return loss, metrics

    def validation_step(self, g):
        p_y_x, p_yp_x, p_yt_x = self(g)

        y = g.ndata['label']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

        metrics = {
            'val/loss': self.cross_entropy_loss(p_yp_x[val_mask], y[val_mask]).item(),
            'val/acc': self.accuracy(pred=p_y_x[val_mask], target=y[val_mask]) * 100,
            'test/acc': self.accuracy(pred=p_y_x[test_mask], target=y[test_mask]) * 100,
        }

        return metrics

    @staticmethod
    def accuracy(pred, target):
        pred = pred.argmax(dim=1) if len(pred.size()) > 1 else pred
        target = target.argmax(dim=1) if len(target.size()) > 1 else target
        return (pred == target).float().mean().item()

    @staticmethod
    def cross_entropy_loss(p_y, y, weighted=False):
        y_onehot = F.one_hot(y.argmax(dim=1))
        loss = -torch.log(p_y + 1e-20) * y_onehot
        loss *= y if weighted else 1
        loss = loss.sum(dim=1).mean()
        return loss
