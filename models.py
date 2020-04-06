# todo make it work for datasets with more than one graph

import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import roc_auc_score
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import add_remaining_self_loops, negative_sampling

from pytorch_lightning import Trainer, LightningModule
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import Adam
from torch_geometric.nn import Node2Vec
from datasets import load_dataset, EdgeSplit
from gnn import GCN


def one_bit_response(x, epsilon, alpha, delta, priv_dim=-1):
    if priv_dim == -1:
        priv_dim = x.size(1)
    exp = math.exp(epsilon)
    x_priv = x[:, :priv_dim]
    p = (x_priv - alpha[:priv_dim]) / delta[:priv_dim]
    p[torch.isnan(p)] = 0.  # nan happens when alpha = beta, so also data.x = alpha, so the prev fraction must be 0
    p = p * (exp - 1) / (exp + 1) + 1 / (exp + 1)
    x_priv = Bernoulli(p).sample()
    x = torch.cat([x_priv, x[:, priv_dim:]], dim=1)
    return x


def transform_features(data, feature, priv_dim=0, epsilon=0):
    if feature == 'priv':
        data.x = one_bit_response(data.x, epsilon, data.alpha, data.delta, priv_dim)
    elif feature == 'locd':
        num_nodes = data.num_nodes
        data.x = None
        data.num_nodes = num_nodes
        data = LocalDegreeProfile()(data)
    return data


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1), device=pos_edge_index.device).float()
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def aggregate_link_prediction_results(outputs):
    link_labels = torch.stack([x['labels'] for x in outputs])
    link_logits = torch.stack([x['logits'] for x in outputs])
    link_probs = torch.sigmoid(link_logits)
    auc = roc_auc_score(link_labels.cpu().numpy().ravel(), link_probs.cpu().numpy().ravel())
    loss = binary_cross_entropy_with_logits(link_logits, link_labels).item()
    logs = {'val_loss': loss, 'val_auc': auc}
    return {'val_loss': loss, 'log': logs, 'progress_bar': logs}


class LitNode2Vec(LightningModule):
    def __init__(self, dataset, embedding_dim, walk_length, context_size, walks_per_node, batch_size,
                 lr=0.01, weight_decay=0):
        super().__init__()
        self.dataset = dataset
        self.model = Node2Vec(dataset[0].num_nodes, embedding_dim, walk_length, context_size, walks_per_node)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, subset):
        return self.model(subset)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, idx):
        edge_index, subset = batch
        loss = self.model.loss(edge_index, subset)
        return {'loss': loss}

    def train_dataloader(self):
        # noinspection PyTypeChecker
        sampler = DataLoader(torch.arange(self.dataset[0].num_nodes), batch_size=self.batch_size, shuffle=True)
        return [(self.dataset[0].edge_index, subset) for subset in sampler]


class Node2VecClassifier(LitNode2Vec):
    def test_dataloader(self):
        return self.dataset

    def test_step(self, data, idx):
        nodes = torch.arange(data.num_nodes).type_as(data.edge_index)
        z = self.model(nodes)
        acc = self.model.test(
            z[data.train_mask], data.y[data.train_mask],
            z[data.test_mask], data.y[data.test_mask], max_iter=150
        )
        return {'val_acc': acc.item()}

    def test_epoch_end(self, outputs):
        avg_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
        logs = {'test_acc': avg_acc}
        return {'avg_test_acc': avg_acc, 'log': logs, 'progress_bar': logs}


class Node2VecLinkPredictor(LitNode2Vec):
    def get_link_logits(self, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = self(total_edge_index[0])
        x_i = self(total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)

    def validation_step(self, data, index):
        pos_edge_index, neg_edge_index = data.val_pos_edge_index, data.val_neg_edge_index
        link_logits = self.get_link_logits(pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def val_dataloader(self):
        return self.dataset

    def test_dataloader(self):
        return self.dataset

    def validation_epoch_end(self, outputs):
        return aggregate_link_prediction_results(outputs)

    def test_step(self, data, index):
        pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        link_logits = self.get_link_logits(pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        auc = result['log']['val_auc']
        result['log']['test_result'] = auc
        return result


class GCNClassifier(LightningModule):
    def __init__(self, dataset, feature, hidden_dim=16, priv_dim=0, epsilon=0, dropout=0.5, lr=0.01, weight_decay=5e-4):
        super().__init__()
        self.dataset = dataset
        self.feature = feature
        self.epsilon = epsilon
        self.priv_dim = priv_dim
        self.lr = lr
        self.weight_decay = weight_decay
        input_dim = 5 if feature == 'locd' else dataset.num_node_features
        self.gcn = GCN(
            input_dim=input_dim,
            output_dim=dataset.num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            priv_input_dim=priv_dim,
            epsilon=epsilon,
            alpha=dataset[0].alpha,
            delta=dataset[0].delta
        )

    def forward(self, data):
        return self.gcn(data.x, data.edge_index)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def prepare_data(self):
        data = transform_features(self.dataset[0], self.feature, self.priv_dim, self.epsilon)
        self.dataset = [data]

    def train_dataloader(self):
        return self.dataset

    def training_step(self, data, index):
        out = self(data)
        loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
        return {'loss': loss}

    def val_dataloader(self):
        return [self.dataset]

    def evaluate(self, data, mask):
        out = self(data)
        loss = cross_entropy(out[mask], data.y[mask])
        pred = out.argmax(dim=1)
        corrects = (pred[mask] == data.y[mask]).sum()
        num_nodes = mask.sum()
        return {'val_loss': loss, 'corrects': corrects, 'num_nodes': num_nodes}

    def validation_step(self, data, index):
        return self.evaluate(data, data.val_mask)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        total_corrects = torch.stack([x['corrects'] for x in outputs]).sum().item()
        total_nodes = torch.stack([x['num_nodes'] for x in outputs]).sum().item()
        avg_acc = total_corrects / total_nodes
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'log': logs, 'progress_bar': logs}

    def test_dataloader(self):
        return [self.dataset]

    def test_step(self, data, index):
        return self.evaluate(data, data.test_mask)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        acc = result['log']['val_acc']
        result['log']['test_result'] = acc
        return result


class GCNLinkPredictor(LightningModule):
    def __init__(self, dataset, feature, hidden_dim=128, output_dim=64, priv_dim=0, epsilon=0,
                 dropout=0, lr=0.01, weight_decay=5e-4):
        super().__init__()
        self.dataset = dataset
        self.feature = feature
        self.epsilon = epsilon
        self.priv_dim = priv_dim
        self.lr = lr
        self.weight_decay = weight_decay
        input_dim = 5 if feature == 'locd' else dataset.num_node_features
        self.gcn = GCN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            priv_input_dim=priv_dim,
            epsilon=epsilon,
            alpha=dataset[0].alpha,
            delta=dataset[0].delta
        )

    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        x = self.gcn(x, pos_edge_index)
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn(x, edge_index)
        return x

    def prepare_data(self):
        data = transform_features(self.dataset[0], self.feature, self.priv_dim, self.epsilon)
        self.dataset = [data]

    def train_dataloader(self):
        # return DataLoader(self.dataset)
        return self.dataset

    def val_dataloader(self):
        return [self.dataset]

    def test_dataloader(self):
        return [self.dataset]

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, data, index):
        x, pos_edge_index = data.x, data.train_pos_edge_index
        pos_edge_index_with_self_loops, _ = add_remaining_self_loops(pos_edge_index, num_nodes=x.size(0))

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1))

        link_logits = self.get_link_logits(x, pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = binary_cross_entropy_with_logits(link_logits, link_labels)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, data, index):
        pos_edge_index, neg_edge_index = data.val_pos_edge_index, data.val_neg_edge_index
        link_logits = self.get_link_logits(data.x, pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def validation_epoch_end(self, outputs):
        return aggregate_link_prediction_results(outputs)

    def test_step(self, data, index):
        pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        link_logits = self.get_link_logits(data.x, pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        auc = result['log']['val_auc']
        result['log']['test_result'] = auc
        return result


def main():
    dataset = load_dataset(
        dataset_name='cora',
        transform=EdgeSplit()
    )
    model = GCNLinkPredictor(dataset, 'priv', priv_dim=dataset.num_node_features, epsilon=3)
    trainer = Trainer(gpus=1, max_epochs=1000, check_val_every_n_epoch=20, checkpoint_callback=False,
                      early_stop_callback=False)
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    main()
