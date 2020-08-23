from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, TrainResult, EvalResult
from pytorch_lightning.metrics.functional import accuracy, average_precision, auroc
from torch.optim import Adam
from torch_geometric.nn import GCNConv, VGAE, APPNP


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False, normalize=True):
        super().__init__()
        # self.ppr = APPNP(K=10, alpha=0.05, add_self_loops=True)
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=not inductive, normalize=normalize)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=not inductive, normalize=normalize)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # x = self.ppr(x, edge_index, edge_weight)
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.log_softmax(x, dim=1)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False, normalize=True):
        super().__init__()
        self.conv = GCNConv(input_dim, hidden_dim, cached=not inductive, normalize=normalize)
        self.conv_mu = GCNConv(hidden_dim, output_dim, cached=not inductive, normalize=normalize)
        self.conv_logvar = GCNConv(hidden_dim, output_dim, cached=not inductive, normalize=normalize)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = F.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv_mu(x, edge_index, edge_weight), self.conv_logvar(x, edge_index, edge_weight)


class NodeClassifier(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden-dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0)
        parser.add_argument('--patience', type=int, default=20)
        return parser

    def __init__(self, hidden_dim=16, dropout=0.5, learning_rate=0.001, weight_decay=0, log_learning_curve=False,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
                normalize=not dataset.use_gdc
            )

    def forward(self, data):
        return self.gcn(data.x, data.edge_index, data.edge_attr)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, data, index):
        out = self(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
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
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
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


class LinkPredictor(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder-hidden-dim', type=int, default=32)
        parser.add_argument('--encoder-output-dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0.0)
        parser.add_argument('--check-val-every-n-epoch', type=int, default=5)
        parser.add_argument('--patience', type=int, default=10)
        return parser

    def __init__(self, encoder_hidden_dim=32, encoder_output_dim=16, dropout=0, learning_rate=0.001, weight_decay=0,
                 log_learning_curve=False, **kwargs):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.log_learning_curve = log_learning_curve
        self.model = None

    def setup(self, stage):
        if stage == 'fit':
            dataset = self.trainer.datamodule
            encoder = GraphEncoder(
                input_dim=dataset.num_features,
                hidden_dim=self.encoder_hidden_dim,
                output_dim=self.encoder_output_dim,
                dropout=self.dropout,
                normalize=not dataset.use_gdc
            )
            self.model = VGAE(encoder=encoder)

    def forward(self, data):
        # Todo fix edge_attr for train/val/test
        x = self.model.encode(data.x, data.train_pos_edge_index)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def model_loss(self, data, pos_edge_index):
        out = self(data)
        loss = self.model.recon_loss(out, pos_edge_index)
        return loss + (1 / data.num_nodes) * self.model.kl_loss()

    def model_test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.model.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        return auroc(target=y, pred=pred), average_precision(target=y, pred=pred)

    def training_step(self, data, index):
        loss = self.model_loss(data, data.train_pos_edge_index)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=False, logger=self.log_learning_curve, on_step=False, on_epoch=True)
        return result

    def validation_step(self, data, index):
        out = self(data)
        loss = self.model_loss(data, data.val_pos_edge_index)
        auc, ap = self.model_test(out, data.val_pos_edge_index, data.val_neg_edge_index)
        result = EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log_dict(
            {'val_loss': loss, 'val_auc': auc, 'val_ap': ap},
            prog_bar=True, logger=self.log_learning_curve, on_step=False, on_epoch=True
        )
        return result

    def test_step(self, data, index):
        out = self(data)
        auc, ap = self.model_test(out, data.test_pos_edge_index, data.test_neg_edge_index)
        result = EvalResult()
        result.log_dict({'test_auc': auc, 'test_ap': ap}, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return result
