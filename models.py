from abc import ABC

import torch
from sklearn.metrics import roc_auc_score
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import add_remaining_self_loops, negative_sampling
from tqdm import trange

from gnn import GCN


class Model(torch.nn.Module):
    def train_model(self, dataloader, optimizer, epochs):
        raise NotImplementedError

    def evaluate(self, dataloader):
        raise NotImplementedError


class Node2VecModel(Node2Vec, Model, ABC):
    def __init__(self, *args, batch_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def train_model(self, dataloader, optimizer, epochs):
        self.train()
        for _ in trange(epochs, desc='Epoch', leave=False):
            for data in dataloader:
                nodes = torch.arange(data.num_nodes, device=data.edge_index.device)
                # noinspection PyTypeChecker
                node_sampler = DataLoader(nodes, batch_size=self.batch_size, shuffle=True)
                for subset in node_sampler:
                    optimizer.zero_grad()
                    loss = self.loss(data.edge_index, subset)
                    loss.backward()
                    optimizer.step()


class GCNClassifier(GCN, Model):
    def train_model(self, dataloader, optimizer, epochs):
        self.train()
        for _ in trange(epochs, desc='Epoch', leave=False):
            for batch in dataloader:
                if batch.train_mask.any():
                    optimizer.zero_grad()
                    out = self(batch.x, batch.edge_index)
                    loss = cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                    loss.backward()
                    optimizer.step()

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        total_nodes, total_corrects = 0, 0
        for batch in dataloader:
            pred = self(batch.x, batch.edge_index).argmax(dim=1)
            total_corrects += (pred[batch.test_mask] == batch.y[batch.test_mask]).sum().item()
            total_nodes += batch.test_mask.sum().item()

        acc = total_corrects / total_nodes
        return acc


class Node2VecClassifier(Node2VecModel):
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        total_nodes, total_corrects = 0, 0
        for data in dataloader:
            z = self(torch.arange(data.num_nodes, device=data.edge_index.device))
            acc = self.test(z[data.train_mask], data.y[data.train_mask],
                            z[data.test_mask], data.y[data.test_mask], max_iter=150)
            n_nodes = data.test_mask.sum().item()
            total_corrects += acc * n_nodes
            total_nodes += n_nodes
        return total_corrects / total_nodes


class LinkPredictor(Model, ABC):
    @staticmethod
    def get_link_labels(pos_edge_index, neg_edge_index):
        link_labels = torch.zeros(pos_edge_index.size(1) +
                                  neg_edge_index.size(1), device=pos_edge_index.device).float()
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        all_labels, all_probs = None, None

        for data in dataloader:
            pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
            link_logits = self.get_link_logits(data.x, pos_edge_index, neg_edge_index)
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            link_probs = torch.sigmoid(link_logits)
            all_labels = link_labels if all_labels is None else torch.cat([all_labels, link_labels])
            all_probs = link_probs if all_probs is None else torch.cat([all_probs, link_probs])

        all_probs = all_probs.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        return roc_auc_score(all_labels, all_probs)


class GCNLinkPredictor(GCN, LinkPredictor):

    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        x = self(x, pos_edge_index)
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)

    def train_model(self, dataloader, optimizer, epochs):
        self.train()
        for _ in trange(epochs, desc='Epoch', leave=False):
            for data in dataloader:
                optimizer.zero_grad()
                x, pos_edge_index = data.x, data.train_pos_edge_index
                pos_edge_index_with_self_loops, _ = add_remaining_self_loops(pos_edge_index, num_nodes=x.size(0))

                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                    num_neg_samples=pos_edge_index.size(1))

                link_logits = self.get_link_logits(x, pos_edge_index, neg_edge_index)
                link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
                loss = binary_cross_entropy_with_logits(link_logits, link_labels)
                loss.backward()
                optimizer.step()


class Node2VecLinkPredictor(Node2VecModel, LinkPredictor):
    def get_link_logits(self, x, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = self(total_edge_index[0])
        x_i = self(total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)
