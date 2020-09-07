from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, TrainResult, EvalResult
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import Adam
from torch_geometric.nn import GCNConv, APPNP, MessagePassing
from torch_geometric.utils import add_remaining_self_loops


class MeanConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=True, **kwargs):
        super().__init__(aggr='mean')
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index, edge_weight=None):
        if self.add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

        x = self.fc(x)
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_j):
        return x_j


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False, normalize=True, k_hop=False):
        super().__init__()
        Conv = (GCNConv if normalize else MeanConv)
        self.appnp = APPNP(K=k_hop, alpha=0, add_self_loops=True) if k_hop else False
        self.conv1 = Conv(input_dim, hidden_dim, cached=not inductive)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=not inductive)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        if self.appnp:
            x = self.appnp(x, edge_index, edge_weight)
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
        parser.add_argument('--k-hop', type=int, default=0)
        parser.add_argument('--patience', type=int, default=50)
        return parser

    def __init__(self, hidden_dim=16, dropout=0.5, learning_rate=0.001, weight_decay=0, appnp=False, normalize=True,
                 log_learning_curve=False, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.appnp = appnp
        self.normalize = normalize
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
                normalize=False,
                k_hop=self.appnp
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
