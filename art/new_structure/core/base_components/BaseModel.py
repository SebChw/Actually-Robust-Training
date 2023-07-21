from typing import Any, Optional

import lightning.pytorch as pl
from datasets import Dataset
from lightning.pytorch.utilities.types import STEP_OUTPUT


class Baseline(pl.LightningModule):
    def validation_step(self, batch, batch_idx) -> float:
        x, y = batch
        y_hat = None  # perform prediction here
        loss = None  # calculate metrics of your choice here
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        raise NotImplementedError


class ClassificationModule(pl.LightningModule):
    """make it so that it can tak any model and perform classification."""
