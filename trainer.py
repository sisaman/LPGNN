import os
import sys
import uuid

import torch
from torch.optim import Adam
from tqdm.auto import tqdm

from utils import colored_text


class Trainer:
    def __init__(
            self,
            max_epochs:     dict(help='maximum number of training epochs') = 500,
            device:         dict(help='desired device for training', choices=['cpu', 'cuda']) = 'cuda',
            checkpoint:     dict(help='use model checkpointing') = True,
            learning_rate: dict(help='learning rate') = 0.01,
            weight_decay: dict(help='weight decay (L2 penalty)') = 0.0,
            logger=None,
    ):
        self.max_epochs = max_epochs
        self.device = device
        self.checkpoint = checkpoint
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logger = logger
        self.model = None

        if checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            self.checkpoint_path = os.path.join('checkpoints', f'{uuid.uuid1()}.pt')

        if not torch.cuda.is_available():
            print(colored_text('CUDA is not available, falling back to CPU', color='red'))
            self.device = 'cpu'

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, model, data):
        self.model = model.to(self.device)
        data = data.to(self.device)
        optimizer = self.configure_optimizers()

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

                if self.logger:
                    self.logger.log({**metrics, 'epoch': epoch})

                if self.checkpoint and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.checkpoint_path)

                # display metrics on progress bar
                epoch_progbar.set_postfix(metrics)
        except KeyboardInterrupt:
            pass

        if self.logger:
            self.logger.log_summary({'val_loss': best_val_loss})

        return best_val_loss

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
