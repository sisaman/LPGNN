from abc import ABC
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=not inductive)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=not inductive)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False):
        super().__init__()
        self.conv = GCNConv(input_dim, hidden_dim, cached=not inductive)
        self.conv_mu = GCNConv(hidden_dim, output_dim, cached=not inductive)
        self.conv_logvar = GCNConv(hidden_dim, output_dim, cached=not inductive)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class BaseModule(LightningModule, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def init_model(self, data): pass

    def fit(self, data, trainer):
        self.init_model(data)
        dataloader = DataLoader([data])
        trainer.fit(self, train_dataloader=dataloader, val_dataloaders=dataloader)

    @staticmethod
    def test(data, trainer, ckpt_path=None):
        dataloader = DataLoader([data])
        trainer.test(test_dataloaders=dataloader, ckpt_path=ckpt_path)

    @staticmethod
    def add_module_specific_args(parent_parser): pass


class NodeClassifier(BaseModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden-dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0)
        parser.add_argument('--min-epochs', type=int, default=0)
        parser.add_argument('--max-epochs', type=int, default=500)
        parser.add_argument('--min-delta', type=float, default=0.0)
        parser.add_argument('--patience', type=int, default=20)
        return parser

    def __init__(self, hidden_dim=16, dropout=0.5, learning_rate=0.001, weight_decay=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.gcn = None

    def init_model(self, data):
        self.gcn = GCN(
            input_dim=data.num_features,
            hidden_dim=self.hidden_dim,
            output_dim=data.num_classes,
            dropout=self.dropout,
            inductive=False
        )

    def forward(self, data):
        return self.gcn(data.x, data.edge_index)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, data, index):
        out = self(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        log = {'loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, data, index):
        out = self(data)
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[data.val_mask], target=data.y[data.val_mask])
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'progress_bar': logs, 'log': logs}

    def test_step(self, data, index):
        out = self(data)
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[data.test_mask], target=data.y[data.test_mask])
        return {'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean().item()
        log = {'test_acc': avg_acc}
        return {'test_acc': avg_acc, 'log': log}


class LinkPredictor(BaseModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder-hidden-dim', type=int, default=32)
        parser.add_argument('--encoder-output-dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0.0)
        parser.add_argument('--min-epochs', type=int, default=0)
        parser.add_argument('--max-epochs', type=int, default=500)
        parser.add_argument('--check-val-every-n-epoch', type=int, default=5)
        parser.add_argument('--min-delta', type=float, default=0.0)
        parser.add_argument('--patience', type=int, default=10)
        return parser

    def __init__(self, encoder_hidden_dim=32, encoder_output_dim=16, dropout=0, learning_rate=0.001, weight_decay=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.model = None

    def init_model(self, data):
        encoder = GraphEncoder(
            input_dim=data.num_features,
            hidden_dim=self.encoder_hidden_dim,
            output_dim=self.encoder_output_dim,
            dropout=self.dropout
        )
        self.model = VGAE(encoder=encoder)

    def forward(self, data):
        x = self.model.encode(data.x, data.train_pos_edge_index)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def model_loss(self, data, pos_edge_index):
        out = self(data)
        loss = self.model.recon_loss(out, pos_edge_index)
        return loss + (1 / data.num_nodes) * self.model.kl_loss()

    def training_step(self, data, index):
        loss = self.model_loss(data, data.train_pos_edge_index)
        log = {'loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, data, index):
        out = self(data)
        loss = self.model_loss(data, data.val_pos_edge_index)
        auc, ap = self.model.test(out, data.val_pos_edge_index, data.val_neg_edge_index)
        return {'val_loss': loss, 'val_auc': auc, 'val_ap': ap}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auc = torch.tensor([x['val_auc'] for x in outputs]).mean()
        avg_ap = torch.tensor([x['val_ap'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_auc': avg_auc, 'val_ap': avg_ap}
        return {'val_loss': avg_loss, 'progress_bar': logs, 'log': logs}

    def test_step(self, data, index):
        out = self(data)
        auc, ap = self.model.test(out, data.test_pos_edge_index, data.test_neg_edge_index)
        return {'test_auc': auc, 'test_ap': ap}

    def test_epoch_end(self, outputs):
        auc = torch.tensor([x['test_auc'] for x in outputs]).mean().item()
        ap = torch.tensor([x['test_ap'] for x in outputs]).mean().item()
        log = {'test_auc': auc, 'test_ap': ap}
        return {'test_auc': auc, 'log': log}
