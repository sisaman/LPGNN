from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, TrainResult, EvalResult
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import Adam
from torch_geometric.nn import GCNConv, VGAE, InnerProductDecoder, GAE
from torch_geometric.transforms import GDC
from torch_geometric.utils import negative_sampling, to_dense_adj


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False, normalize=True):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=not inductive, normalize=normalize)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=not inductive, normalize=normalize)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.log_softmax(x, dim=1)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, inductive=False, normalize=True, variational=False):
        super().__init__()
        self.conv = GCNConv(input_dim, hidden_dim, cached=not inductive, normalize=normalize)
        self.conv_mu = GCNConv(hidden_dim, output_dim, cached=not inductive, normalize=normalize)
        if variational:
            self.conv_logvar = GCNConv(hidden_dim, output_dim, cached=not inductive, normalize=normalize)
        self.dropout = dropout
        self.variational = variational

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = F.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.variational:
            return self.conv_mu(x, edge_index, edge_weight), self.conv_logvar(x, edge_index, edge_weight)
        else:
            return self.conv_mu(x, edge_index, edge_weight)


class AugmentedNodeClassifier(LightningModule):

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden-dim', type=int, default=16)
        parser.add_argument('--encoder-hidden-dim', type=int, default=32)
        parser.add_argument('--encoder-output-dim', type=int, default=16)
        parser.add_argument('--avg-degree', type=int, default=20)
        parser.add_argument('--encoder-dropout', type=float, default=0)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--weight-decay', type=float, default=0)
        parser.add_argument('--patience', type=int, default=20)
        return parser

    def __init__(self, hidden_dim=16, encoder_hidden_dim=32, encoder_output_dim=16, avg_degree=32,
                 dropout=0.5, encoder_dropout=0, learning_rate=0.001, weight_decay=0, log_learning_curve=False,
                 neg_sampling=False, variational=False,  **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.avg_degree = avg_degree
        self.gnn_dropout = dropout
        self.encoder_dropout = encoder_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.neg_sampling = neg_sampling
        self.variational = variational
        self.save_hyperparameters()

        self.log_learning_curve = log_learning_curve
        self.generator = None
        self.discriminator = None
        self.gdc = GDC()

    def init_generator(self, input_dim):
        encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=self.encoder_hidden_dim,
            output_dim=self.encoder_output_dim,
            dropout=self.encoder_dropout,
            inductive=False,
            normalize=True,
            variational=self.variational
        )

        self.generator = (VGAE if self.variational else GAE)(encoder=encoder, decoder=InnerProductDecoder())

    def init_discriminator(self, input_dim, num_classes):
        self.discriminator = GCN(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_classes,
            dropout=self.gnn_dropout,
            inductive=True,
            normalize=True
        )

    def setup(self, stage):
        if stage == 'fit':
            dataset = self.trainer.datamodule
            self.init_generator(input_dim=dataset.num_features)
            self.init_discriminator(input_dim=dataset.num_features, num_classes=dataset.num_classes)

    def forward(self, data, discriminator=True):
        z = self.generator.encode(data.x, data.edge_index)

        if not discriminator:
            return z

        if self.neg_sampling:
            pos_edge_index = data.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=2 * (data.num_nodes * self.avg_degree - data.num_edges),
                # num_neg_samples=data.num_nodes**2 - data.num_edges
            )

            pos_edge_weight = torch.ones(pos_edge_index.size(1), device=pos_edge_index.device)
            neg_edge_weight = self.generator.decoder(z, neg_edge_index)
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_weight = torch.cat([pos_edge_weight, neg_edge_weight])

            new_edge_index, _ = self.gdc.sparsify_sparse(
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=data.num_nodes,
                method='threshold',
                avg_degree=self.avg_degree
            )
        else:
            adj = self.generator.decoder.forward_all(z)
            adj = adj + to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze()
            new_edge_index, _ = self.gdc.sparsify_dense(matrix=adj, method='threshold', avg_degree=self.avg_degree)

        return self.discriminator(data.x, new_edge_index)

    def generator_step(self, data):
        out = self(data, discriminator=False)
        g_loss = self.generator.recon_loss(out, data.edge_index)
        if self.variational:
            g_loss += (1 / data.num_nodes) * self.generator.kl_loss()
        return g_loss

    def discriminator_step(self, data, mask):
        out = self(data, discriminator=True)
        d_loss = F.nll_loss(out[mask], data.y[mask], ignore_index=-1)
        pred = out.argmax(dim=1)
        acc = accuracy(pred=pred[mask], target=data.y[mask])
        return d_loss, acc

    def training_step(self, data, data_idx, optimizer_idx):
        # train generator
        if optimizer_idx == 0:
            g_loss = self.generator_step(data)
            result = TrainResult(minimize=g_loss)
            result.log('train_g_loss', g_loss, prog_bar=True, logger=self.log_learning_curve, on_step=False,
                       on_epoch=True)
            return result

        # train discriminator
        if optimizer_idx == 1:
            d_loss, acc = self.discriminator_step(data, data.train_mask)
            result = TrainResult(minimize=d_loss)
            result.log_dict(
                dictionary={'train_d_loss': d_loss, 'train_acc': acc},
                prog_bar=True, logger=self.log_learning_curve, on_step=False, on_epoch=True
            )
            return result

    def validation_step(self, data, data_idx):
        d_loss, acc = self.discriminator_step(data, data.val_mask)
        result = EvalResult(early_stop_on=d_loss, checkpoint_on=d_loss)
        result.log_dict(
            dictionary={'val_loss': d_loss, 'val_acc': acc},
            prog_bar=True, logger=self.log_learning_curve, on_step=False, on_epoch=True
        )
        return result

    def test_step(self, data, index):
        d_loss, acc = self.discriminator_step(data, data.test_mask)
        result = EvalResult()
        result.log('test_acc', acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return result

    def configure_optimizers(self):
        opt_g = Adam(self.generator.parameters(), lr=self.learning_rate)
        opt_d = Adam(self.discriminator.parameters(), lr=self.learning_rate)
        return [opt_g, opt_d], []
