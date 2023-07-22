from typing import Any

import lightning.pytorch as pl
import torch


class ClassificationModel(pl.LightningModule):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    # TODO we should define templates for tasks, start from classification.

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch["x"], train_batch["y"].unsqueeze(1).float()
        y_hat = self.model(X)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch["x"], val_batch["y"].unsqueeze(1).float()
        y_hat = self.model(X)
        loss = self.loss_fn(y_hat, y)
        self.log("validation_loss", loss)
        # TODO inject metric calculation here
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
