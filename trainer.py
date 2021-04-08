import os
import sys
import uuid

import torch
from torch.optim import SGD, Adam
from tqdm.auto import tqdm

from utils import colored_text


class Trainer:
    def __init__(
            self,
            optimizer:      dict('optimization algorithm', choices=['sgd', 'adam']) = 'sgd',
            max_epochs:     dict(help='maximum number of training epochs') = 500,
            device:         dict(help='desired device for training', choices=['cpu', 'cuda']) = 'cuda',
            checkpoint:     dict(help='use model checkpointing') = True,
            learning_rate:  dict(help='learning rate') = 0.01,
            weight_decay:   dict(help='weight decay (L2 penalty)') = 0.0,
            patience:       dict(help='early-stopping patience window size') = 100,
            logger=None,
    ):
        self.optimizer_name = optimizer
        self.max_epochs = max_epochs
        self.device = device
        self.checkpoint = checkpoint
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.logger = logger
        self.model = None

        if checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            self.checkpoint_path = os.path.join('checkpoints', f'{uuid.uuid1()}.pt')

        if not torch.cuda.is_available():
            print(colored_text('CUDA is not available, falling back to CPU', color='red'))
            self.device = 'cpu'

    def configure_optimizers(self):
        return {
            'sgd': SGD, 'adam': Adam
        }[self.optimizer_name](self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, model, data):
        self.model = model.to(self.device)
        data = data.to(self.device)
        optimizer = self.configure_optimizers()

        num_epochs_without_improvement = 0
        best_val_acc = 0
        best_val_loss = float('inf')
        epoch_progbar = tqdm(range(1, self.max_epochs + 1), desc='Epoch: ', leave=False, position=1, file=sys.stdout)

        try:
            for epoch in epoch_progbar:
                metrics = {}
                train_metrics = self._train(data, optimizer)
                metrics.update(train_metrics)

                val_metrics = self._validation(data)
                metrics.update(val_metrics)
                val_loss = val_metrics['val_loss']
                val_acc = val_metrics['val_acc']

                if self.logger:
                    self.logger.log({**metrics, 'epoch': epoch})

                if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    num_epochs_without_improvement = 0
                    if self.checkpoint:
                        torch.save(self.model.state_dict(), self.checkpoint_path)
                else:
                    num_epochs_without_improvement += 1
                    if num_epochs_without_improvement >= self.patience > 0:
                        break

                # display metrics on progress bar
                epoch_progbar.set_postfix(metrics)
        except KeyboardInterrupt:
            pass

        best_metrics = {'val_loss': best_val_loss, 'val_acc': best_val_acc}

        if self.logger:
            self.logger.log_summary(best_metrics)

        return best_metrics

    def _train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        loss, metrics = self.model.training_step(data)
        loss.backward()
        optimizer.step()
        return metrics

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        return self.model.validation_step(data)

    @torch.no_grad()
    def test(self, data):
        if self.checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.model.eval()
        metrics = self.model.test_step(data)

        if self.logger:
            self.logger.log_summary(metrics)

        return metrics
