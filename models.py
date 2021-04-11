import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import accuracy
from torch.nn import Dropout
from torch_geometric.nn import MessagePassing, SAGEConv, GCNConv
from torch_sparse import matmul


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
                 x_steps:               dict(help='KProp step parameter', option='-k') = 0,
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

        self.lambdaa = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.cached_yt = None

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.x_prop(x, adj_t)
        x = self.gnn(x, adj_t)
        p_y_x = p_yt_x = F.softmax(x, dim=1)            # P(y|x)
        p_yp_x = torch.matmul(p_y_x, data.T)            # P(y'|x')

        if self.propagate_predictions:
            p_yt_x = self.y_prop(p_yp_x, data.adj_t)    # P(y~|x')

        return p_y_x, p_yp_x, p_yt_x

    @staticmethod
    def cross_entropy_loss(p_y, y, weighted=False):
        y_onehot = F.one_hot(y.argmax(dim=1))
        loss = -torch.log(p_y + 1e-20) * y_onehot
        loss *= y if weighted else 1
        loss = loss.sum(dim=1).mean()
        return loss

    def model_loss(self, p_yp, p_yt, yp, yt):
        lambdaa = torch.sigmoid(self.lambdaa)
        loss_p = self.cross_entropy_loss(p_y=p_yp, y=yp)
        loss_t = self.cross_entropy_loss(p_y=p_yt, y=yt)
        loss = lambdaa * loss_p + (1 - lambdaa) * loss_t
        return loss

    def training_step(self, data):
        p_y_x, p_yp_x, p_yt_x = self(data)

        if self.cached_yt is None:
            yp = data.y.float()
            yp[data.test_mask] = 0  # to avoid using test labels
            self.cached_yt = self.y_prop(yp, data.adj_t)  # y~

        loss = self.model_loss(
            p_yp=p_yp_x[data.train_mask],
            p_yt=p_yt_x[data.train_mask],
            yp=data.y[data.train_mask],
            yt=self.cached_yt[data.train_mask],
        )

        metrics = {
            'train/loss': loss.item(),
            'train/acc': accuracy(
                pred=p_y_x[data.train_mask].argmax(dim=1),
                target=self.cached_yt[data.train_mask].argmax(dim=1)
            ) * 100,
            'train/lambda': torch.sigmoid(self.lambdaa).item()
        }

        return loss, metrics

    def validation_step(self, data):
        p_y_x, p_yp_x, p_yt_x = self(data)

        val_loss = self.model_loss(
            p_yp=p_yp_x[data.val_mask],
            p_yt=p_yt_x[data.val_mask],
            yp=data.y[data.val_mask],
            yt=self.cached_yt[data.val_mask],
        )

        metrics = {
            'val/loss': val_loss.item(),
            'val/acc': accuracy(pred=p_y_x[data.val_mask].argmax(dim=1), target=self.cached_yt[data.val_mask].argmax(dim=1)) * 100,
            'val/acc_p': accuracy(pred=p_yp_x[data.val_mask].argmax(dim=1), target=data.y[data.val_mask].argmax(dim=1)) * 100,
            'val/acc_t': accuracy(pred=p_yt_x[data.val_mask].argmax(dim=1), target=self.cached_yt[data.val_mask].argmax(dim=1)) * 100,
            'test/acc': accuracy(pred=p_y_x[data.test_mask].argmax(dim=1), target=data.y[data.test_mask].argmax(dim=1)) * 100,
        }

        return metrics
