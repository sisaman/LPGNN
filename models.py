import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import add_remaining_self_loops, negative_sampling

from datasets import load_dataset, GraphLoader
from gnn import GCN


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1), device=pos_edge_index.device).float()
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def aggregate_link_prediction_results(outputs, metric='loss'):
    link_labels = torch.stack([x['labels'] for x in outputs])
    link_logits = torch.stack([x['logits'] for x in outputs])
    if metric == 'auc':
        link_probs = torch.sigmoid(link_logits)
        result = roc_auc_score(link_labels.cpu().numpy().ravel(), link_probs.cpu().numpy().ravel())
    else:
        result = binary_cross_entropy_with_logits(link_logits, link_labels).item()

    logs = {'val_'+metric: result}
    return {'val_'+metric: result, 'log': logs, 'progress_bar': logs}


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
    def evaluate(self, mask):
        nodes = torch.arange(self.data.num_nodes).type_as(self.data.edge_index)
        z = self.model(nodes)
        acc = self.model.test(
            z[self.data.train_mask], self.data.y[self.data.train_mask],
            z[mask], self.data.y[mask], max_iter=150
        ).item()
        return {'val_acc': acc}

    def val_dataloader(self):
        return GraphLoader(self.data)

    def validation_step(self, data, idx):
        return self.evaluate(data.val_mask)

    def test_dataloader(self):
        return GraphLoader(self.data)

    def test_step(self, data, idx):
        return self.evaluate(data.test_mask)

    def validation_epoch_end(self, outputs):
        avg_acc = torch.tensor([x['val_acc'] for x in outputs]).mean().item()
        logs = {'val_acc': avg_acc}
        return {'avg_val_acc': avg_acc, 'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        acc = result['log']['val_acc']
        result['log']['test_result'] = acc
        return result


class Node2VecLinkPredictor(LitNode2Vec):
    def get_link_logits(self, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = self(total_edge_index[0])
        x_i = self(total_edge_index[1])
        return (x_i * x_j).sum(dim=1)

    def validation_step(self, data, index):
        pos_edge_index, neg_edge_index = data.val_pos_edge_index, data.val_neg_edge_index
        link_logits = self.get_link_logits(pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def val_dataloader(self):
        return GraphLoader(self.data)

    def test_dataloader(self):
        return GraphLoader(self.data)

    def validation_epoch_end(self, outputs):
        return aggregate_link_prediction_results(outputs, 'loss')

    def test_step(self, data, index):
        pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        link_logits = self.get_link_logits(pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def test_epoch_end(self, outputs):
        result = aggregate_link_prediction_results(outputs, 'auc')
        auc = result['log']['val_auc']
        result['log']['test_result'] = auc
        return result


class GCNClassifier(LightningModule):
    def __init__(self, data, hidden_dim=16, epsilon=1, dropout=0.5, lr=0.01, weight_decay=5e-4):
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
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return GraphLoader(self.data)

    def training_step(self, data, index):
        out = self(data)
        loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
        return {'loss': loss}

    def val_dataloader(self):
        return GraphLoader(self.data)

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
        return GraphLoader(self.data)

    def test_step(self, data, index):
        return self.evaluate(data, data.test_mask)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        acc = result['log']['val_acc']
        result['log']['test_result'] = acc
        return result


class GCNLinkPredictor(LightningModule):
    def __init__(self, data, hidden_dim=128, output_dim=64, epsilon=1, dropout=0, lr=0.01, weight_decay=0):
        super().__init__()
        self.data = data
        self.lr = lr
        self.weight_decay = weight_decay
        self.gcn = GCN(
            input_dim=data.num_node_features,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epsilon=epsilon,
            alpha=data.alpha,
            delta=data.delta
        )

    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        x = self.gcn(x, pos_edge_index)
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return (x_i * x_j).sum(dim=1)

    def forward(self, data):
        x = self.gcn(data.x, data.edge_index, data.priv_mask)
        return x

    def train_dataloader(self):
        return GraphLoader(self.data)

    def val_dataloader(self):
        return GraphLoader(self.data)

    def test_dataloader(self):
        return GraphLoader(self.data)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, data, index):
        x, pos_edge_index = data.x, data.train_pos_edge_index
        pos_edge_index_with_self_loops, _ = add_remaining_self_loops(pos_edge_index, num_nodes=data.num_nodes)

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=data.num_nodes,
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
        return aggregate_link_prediction_results(outputs, 'loss')

    def test_step(self, data, index):
        pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        link_logits = self.get_link_logits(data.x, pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        return {'labels': link_labels, 'logits': link_logits}

    def test_epoch_end(self, outputs):
        result = aggregate_link_prediction_results(outputs, 'auc')
        auc = result['log']['val_auc']
        result['log']['test_result'] = auc
        return result


def main():
    dataset = load_dataset(
        dataset_name='cora',
        # task_name='linkpred'
        # transform=EdgeSplit()
    ).to('cuda')
    model = GCNClassifier(dataset, 16)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=True, mode='min')
    for i in range(1):
        trainer = Trainer(gpus=1, max_epochs=500, check_val_every_n_epoch=20, checkpoint_callback=False,
                          early_stop_callback=early_stop_callback)
        trainer.fit(model)
        trainer.test()


if __name__ == '__main__':
    main()
