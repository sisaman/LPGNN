from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
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
        # x = self.bn(x)
        x = F.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class NodeClassifier(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden-dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0)
        parser.add_argument('--min-epochs', type=int, default=10)
        parser.add_argument('--max-epochs', type=int, default=500)
        parser.add_argument('--min-delta', type=float, default=0.0)
        parser.add_argument('--patience', type=int, default=20)
        return parser

    def __init__(self, num_classes, input_dim, hidden_dim=16,
                 dropout=0.5, learning_rate=0.001, weight_decay=0, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.model = GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            dropout=dropout,
            inductive=False
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, data, index):
        out = self(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        return {'loss': loss}

    def validation_step(self, data, index):
        out = self(data)
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'progress_bar': logs}

    def test_step(self, data, index):
        out = self(data)
        pred = out.argmax(dim=1)
        corrects = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        num_test_nodes = data.test_mask.sum().item()
        return {'corrects': corrects, 'num_nodes': num_test_nodes}

    def test_epoch_end(self, outputs):
        total_corrects = sum([x['corrects'] for x in outputs])
        total_nodes = sum([x['num_nodes'] for x in outputs])
        acc = total_corrects / total_nodes
        log = {'test_result': acc}
        return {'test_acc': acc, 'log': log}


class LinkPredictor(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder-hidden-dim', type=int, default=32)
        parser.add_argument('--encoder-output-dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0.0)
        parser.add_argument('--min-epochs', type=int, default=100)
        parser.add_argument('--max-epochs', type=int, default=500)
        parser.add_argument('--check-val-every-n-epoch', type=int, default=10)
        parser.add_argument('--min-delta', type=float, default=0.0)
        parser.add_argument('--patience', type=int, default=10)
        return parser

    def __init__(self, input_dim, encoder_hidden_dim=32, encoder_output_dim=16,
                 dropout=0, learning_rate=0.001, weight_decay=0, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.model = VGAE(encoder=encoder)
        self.trainer = None

    def forward(self, data):
        x = self.model.encode(data.x, data.train_pos_edge_index)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def model_loss(self, data, pos_edge_index):
        z = self(data)
        loss = self.model.recon_loss(z, pos_edge_index)
        return loss + (1 / data.num_nodes) * self.model.kl_loss()

    def training_step(self, data, index):
        loss = self.model_loss(data, data.train_pos_edge_index)
        return {'loss': loss}

    def validation_step(self, data, index):
        loss = self.model_loss(data, data.val_pos_edge_index)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        log = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'progress_bar': log}

    def test_step(self, data, index):
        z = self(data)
        auc, ap = self.model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        return {'auc': auc, 'ap': ap}

    def test_epoch_end(self, outputs):
        auc = torch.tensor([x['auc'] for x in outputs]).mean().item()
        log = {'test_result': auc}
        return {'test_auc': auc, 'log': log}
