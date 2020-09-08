from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, TrainResult, EvalResult
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import Adam
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops


class KProp(MessagePassing):
    def __init__(self, K: int, cached=False, aggr='gcn'):
        self.K = K
        self.add_self_loops = K == 1
        self.cached = cached
        self._cache = None
        self.agg = aggr
        super().__init__(aggr='add' if aggr == 'gcn' else 'mean')

    def forward(self, x, edge_index, edge_weight=None):
        if self._cache is None:
            if self.agg == 'gcn':
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, x.size(self.node_dim),
                    add_self_loops=self.add_self_loops, dtype=x.dtype
                )
            else:
                edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
                if self.add_self_loops:
                    edge_index, edge_weight = add_remaining_self_loops(
                        edge_index, edge_weight, num_nodes=x.size(self.node_dim)
                    )
            if self.cached:
                self._cache = edge_index, edge_weight
        else:
            edge_index, edge_weight = self._cache

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return x

    # noinspection PyMethodOverriding
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False, aggr='gcn', hops=1):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, hidden_dim)
        self.conv1 = KProp(K=hops, cached=not inductive, aggr=aggr)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=not inductive)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.fc(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.log_softmax(x, dim=1)
        return x


class NodeClassifier(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden-dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--learning-rate', '--lr', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0)
        parser.add_argument('-k', '--hops', type=int, default=1)
        parser.add_argument('--aggr', type=str, default='gcn')
        return parser

    def __init__(self, hidden_dim=16, dropout=0.5, learning_rate=0.001, weight_decay=0, hops=1, aggr='gcn',
                 log_learning_curve=False, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hops = hops
        self.aggr = aggr
        self.save_hyperparameters()
        self.log_learning_curve = log_learning_curve
        self.gcn = None

    def setup(self, stage):
        if stage == 'fit':
            dataset = self.trainer.datamodule
            self.gcn = GCN(
                input_dim=dataset.num_features,
                hidden_dim=self.hidden_dim,
                output_dim=dataset.num_classes,
                dropout=self.dropout,
                inductive=False,
                hops=self.hops,
                aggr=self.aggr
            )

    def forward(self, data):
        return self.gcn(data.x, data.edge_index, data.edge_attr)

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
