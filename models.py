import logging

import torch
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import VGAE

from datasets import load_dataset, GraphLoader
from gnn import GCN, GraphEncoder, GraphSAGE
from mechanisms import privatize

logging.disable(logging.INFO)


class GCNClassifier(LightningModule):
    def __init__(self, data, hidden_dim=16, dropout=0.5, lr=0.01, weight_decay=5e-4):
        super().__init__()
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.gnn = GCN(
            input_dim=data.num_node_features,
            output_dim=data.num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, data):
        return self.gnn(data.x, data.edge_index)

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
        return {'loss': loss}

    def validation_step(self, data, index):
        out = self(data)
        loss = cross_entropy(out[data.val_mask], data.y[data.val_mask])
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


class VGAELinkPredictor(LightningModule):
    def __init__(self, data, output_dim=16, lr=0.01, weight_decay=0.0):
        super().__init__()
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay

        encoder = GraphEncoder(
            input_dim=data.num_node_features,
            output_dim=output_dim
        )

        self.model = VGAE(encoder=encoder)

    def forward(self, data):
        x = self.model.encode(data.x, data.train_pos_edge_index)
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
        return {'val_loss': avg_loss, 'progress_bar': log}

    def test_step(self, data, index):
        z = self(data)
        auc, ap = self.model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        return {'auc': auc, 'ap': ap}

    def test_epoch_end(self, outputs):
        auc = torch.tensor([x['auc'] for x in outputs]).mean().item()
        log = {'test_result': auc}
        return {'test_auc': auc, 'log': log}


class GraphSAGEClassifier(GCNClassifier):
    def __init__(self, data, hidden_dim=16, dropout=0.5, lr=0.01, weight_decay=5e-4):
        super().__init__(data, hidden_dim, dropout, lr, weight_decay)
        self.gcn = GraphSAGE(
            input_dim=data.num_node_features,
            output_dim=data.num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )


def main():
    seed_everything(12345)

    data = load_dataset(
        dataset_name='pubmed',
        # split_edges=True
    ).to('cuda')

    data = privatize(data, rfr=0.2, pfr=0., eps=3, method='bit')

    for i in range(1):
        print('RUN', i)
        model = GCNClassifier(data, lr=.01, weight_decay=0.001, dropout=0.5)
        # model = GraphSAGEClassifier(data, lr=.01, weight_decay=0.01, dropout=0.5)
        # model = VGAELinkPredictor(data, lr=0.01, weight_decay=0.001)

        # noinspection PyTypeChecker
        trainer = Trainer(
            gpus=1, max_epochs=500, checkpoint_callback=False,
            # early_stop_callback=EarlyStopping(monitor='val_loss', min_delta=0, patience=20),
            early_stop_callback=False,
            weights_summary=None,
            # min_epochs=100,
            logger=False,
            # check_val_every_n_epoch=10
        )

        # lr_finder = trainer.lr_find(model)
        # print(lr_finder.suggestion())
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        trainer.fit(model)
        trainer.test()


if __name__ == '__main__':
    main()
