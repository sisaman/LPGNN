from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, TrainResult, EvalResult
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import Linear, Dropout
from torch.optim import Adam
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul


class KProp(MessagePassing):
    def __init__(self, in_channels, out_channels, K, aggregator, add_self_loops, cached=False):
        super().__init__(aggr='add' if aggregator == 'gcn' else aggregator)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = Linear(in_channels, out_channels)
        self.K = K
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


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, K, aggregator, self_loops):
        super().__init__()
        self.conv1 = KProp(input_dim, hidden_dim, K=K, aggregator=aggregator, add_self_loops=self_loops, cached=True)
        self.conv2 = KProp(hidden_dim, output_dim, K=1, aggregator=aggregator, add_self_loops=True, cached=False)
        self.dropout = Dropout(p=dropout)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = torch.selu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        x = F.log_softmax(x, dim=1)
        return x


class NodeClassifier(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden-dim', type=int, default=16)
        parser.add_argument('--dropout', '--dp', type=float, default=0)
        parser.add_argument('--learning-rate', '--lr', type=float, default=0.001)
        parser.add_argument('--weight-decay', '--wd', type=float, default=0)
        parser.add_argument('--self-loops', action='store_true', default=False)
        return parser

    def __init__(self, hidden_dim=16, dropout=0.5, learning_rate=0.001, weight_decay=0, K=1, aggregator='gcn',
                 self_loops=True, log_learning_curve=False, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = K
        self.aggregator = aggregator
        self.self_loops = self_loops
        self.save_hyperparameters()
        self.log_learning_curve = log_learning_curve
        self.gcn = None

    def setup(self, stage):
        if stage == 'fit':
            dataset = self.trainer.datamodule
            self.gcn = GNN(
                input_dim=dataset.num_features,
                hidden_dim=self.hidden_dim,
                output_dim=dataset.num_classes,
                dropout=self.dropout,
                K=self.steps,
                aggregator=self.aggregator,
                self_loops=self.self_loops
            )

    def forward(self, data):
        return self.gcn(data.x, data.adj_t)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, data, index):
        out = self(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], ignore_index=-1)
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[data.train_mask], target=data.y[data.train_mask])
        result = TrainResult(minimize=loss)
        result.log_dict(
            dictionary={'train_loss': loss, 'train_acc': acc},
            prog_bar=True, logger=self.log_learning_curve, on_step=False, on_epoch=True
        )
        return result

    def validation_step(self, data, index):
        out = self(data)
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask], ignore_index=-1)
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[data.val_mask], target=data.y[data.val_mask])
        result = EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log_dict(
            dictionary={'val_loss': loss, 'val_acc': acc},
            prog_bar=True, logger=self.log_learning_curve, on_step=False, on_epoch=True
        )
        return result

    def test_step(self, data, index):
        out = self(data)
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[data.test_mask], target=data.y[data.test_mask])
        result = EvalResult()
        result.log('test_acc', acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return result
