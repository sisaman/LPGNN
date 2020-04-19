import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only
from sklearn.metrics import roc_auc_score
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec, VGAE
from datasets import load_dataset, GraphLoader, privatize
from gnn import GCN, GraphEncoder
import logging
logging.disable(logging.INFO)


class ResultLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.result = None

    @property
    def experiment(self):
        return self

    @rank_zero_only
    def log_metrics(self, metrics, step=None): pass

    def set_result(self, result):
        self.result = result

    @rank_zero_only
    def log_hyperparams(self, parameters): pass

    @property
    def name(self): return 'ResultLogger'

    @property
    def version(self): return 0.1


class LitNode2Vec(LightningModule):
    def __init__(self, data, embedding_dim, walk_length, context_size, walks_per_node, batch_size,
                 lr=0.01, weight_decay=0):
        super().__init__()
        self.data = data
        self.model = Node2Vec(self.data.num_nodes, embedding_dim, walk_length, context_size, walks_per_node)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, subset):
        return self.model(subset)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, subset, idx):
        loss = self.model.loss(self.data.edge_index, subset)
        return {'loss': loss}

    def train_dataloader(self):
        # noinspection PyTypeChecker
        return DataLoader(torch.arange(self.data.num_nodes), batch_size=self.batch_size, shuffle=True)


class Node2VecClassifier(LitNode2Vec):
    def test_dataloader(self):
        return GraphLoader(self.data)

    def test_step(self, data, idx):
        nodes = torch.arange(self.data.num_nodes).type_as(self.data.edge_index)
        z = self.model(nodes)
        acc = self.model.test(
            z[self.data.train_mask], self.data.y[self.data.train_mask],
            z[data.test_mask], self.data.y[data.test_mask], max_iter=150
        ).item()
        return {'val_acc': acc}

    def test_epoch_end(self, outputs):
        acc = torch.tensor([x['val_acc'] for x in outputs]).mean().item()
        logs = {'test_acc': acc}
        self.logger.set_result(acc)
        return {'test_acc': acc, 'log': logs, 'progress_bar': logs}


class Node2VecLinkPredictor(LitNode2Vec):
    @staticmethod
    def get_link_labels(pos_edge_index, neg_edge_index):
        link_labels = torch.zeros(pos_edge_index.size(1) +
                                  neg_edge_index.size(1), device=pos_edge_index.device).float()
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def get_link_logits(self, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = self(total_edge_index[0])
        x_i = self(total_edge_index[1])
        return (x_i * x_j).sum(dim=1)

    def test_dataloader(self):
        return GraphLoader(self.data)

    def test_step(self, data, index):
        pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        link_logits = self.get_link_logits(pos_edge_index, neg_edge_index)
        link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def test_epoch_end(self, outputs):
        link_labels = torch.stack([x['labels'] for x in outputs])
        link_logits = torch.stack([x['logits'] for x in outputs])
        link_probs = torch.sigmoid(link_logits)
        auc = roc_auc_score(link_labels.cpu().numpy().ravel(), link_probs.cpu().numpy().ravel())
        logs = {'test_auc': auc}
        self.logger.set_result(auc)
        return {'test_auc': auc, 'log': logs, 'progress_bar': logs}


class GCNClassifier(LightningModule):
    def __init__(self, data, hidden_dim=16, epsilon=1.0, dropout=0.5, lr=0.01, weight_decay=5e-4):
        super().__init__()
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.gcn = GCN(
            input_dim=data.num_node_features,
            output_dim=data.num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epsilon=epsilon,
            alpha=data.alpha,
            delta=data.delta
        )

    def forward(self, data):
        return self.gcn(data.x, data.edge_index, data.priv_mask)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return GraphLoader(self.data)

    def val_dataloader(self):
        return GraphLoader(self.data)

    def test_dataloader(self):
        return GraphLoader(self.data)

    def training_step(self, data, index):
        out = self(data)
        loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, data, index):
        out = self(data)
        loss = cross_entropy(out[data.val_mask], data.y[data.val_mask])
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

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
        log = {'test_acc': acc}
        self.logger.set_result(acc)
        return {'test_acc': acc, 'log': log, 'progress_bar': log}


class VGAELinkPredictor(LightningModule):
    def __init__(self, data, output_dim=16, epsilon=1.0, lr=0.01, weight_decay=0.0):
        super().__init__()
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay

        encoder = GraphEncoder(
            input_dim=data.num_node_features,
            output_dim=output_dim,
            epsilon=epsilon,
            alpha=data.alpha,
            delta=data.delta
        )

        self.model = VGAE(encoder=encoder)

    def forward(self, data):
        x = self.model.encode(data.x, data.train_pos_edge_index, data.priv_mask)
        return x

    def train_dataloader(self):
        return GraphLoader(self.data)

    def val_dataloader(self):
        return GraphLoader(self.data)

    def test_dataloader(self):
        return GraphLoader(self.data)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
        return {'val_loss': avg_loss, 'log': log, 'progress_bar': log}

    def test_step(self, data, index):
        z = self(data)
        auc, ap = self.model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        return {'auc': auc, 'ap': ap}

    def test_epoch_end(self, outputs):
        auc = torch.tensor([x['auc'] for x in outputs]).mean().item()
        ap = torch.tensor([x['ap'] for x in outputs]).mean().item()
        log = {'test_auc': auc, 'test_ap': ap}
        self.logger.set_result(auc)
        return {'test_auc': auc, 'test_ap': ap, 'log': log, 'progress_bar': log}


def main():
    torch.manual_seed(12345)

    dataset = load_dataset(
        dataset_name='bitcoin',
        split_edges=True
    ).to('cuda')

    eps = 9
    dataset = privatize(dataset, pnr=1, pfr=1, eps=eps, method='bit')

    for i in range(10):
        print('RUN', i)
        # model = GCNClassifier(dataset, lr=.01, weight_decay=0.0001, dropout=0.5, epsilon=eps)
        model = VGAELinkPredictor(dataset, lr=0.001, weight_decay=0.0001)

        # noinspection PyTypeChecker
        trainer = Trainer(gpus=1, max_epochs=500, checkpoint_callback=False,
                          early_stop_callback=EarlyStopping(monitor='val_loss', min_delta=0, patience=10),
                          # early_stop_callback=False,
                          weights_summary=None,
                          min_epochs=100,
                          logger=ResultLogger(),
                          check_val_every_n_epoch=10
                          )
        trainer.fit(model)
        trainer.test()


if __name__ == '__main__':
    main()
