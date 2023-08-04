from typing import Any, Optional

import lightning.pytorch as pl
import torch.nn
from lightning.pytorch.utilities.types import STEP_OUTPUT

from art.metric_calculator import MetricCalculator


class Baseline(pl.LightningModule):
    def __init__(self, accelerator="cpu"):
        super().__init__()
        self.accelerator = accelerator

    def validation_step(self, batch, batch_idx) -> float:
        x, y = batch
        y_hat = None  # perform prediction here
        loss = None  # calculate metrics of your choice here
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        raise NotImplementedError

    # TODO think if this shouldn't be added in some class higher in hierarchy like Model
    def set_metric(self, metric):
        self.metric = metric


class ArtModule(pl.LightningModule):
    def __init__(
        self,
        device="cpu",
    ):
        super().__init__()
        self.regularized = True
        self.device.type = device
        self.metric_calculator = MetricCalculator()

    """
    I think in case of models we can easily write some general purpose function that will do the job.

    But maybe this should be specific to the Stage we are currently at?

    1. Turn of weight decay if turned on -> easy to do in optimizer.
    2. Don't use fancy optimizers -> stick to Adam (But maybe this depends on stage?)
    3. Turn off learning rate decays -> Again, maybe this depends on stage?
    4. Set all dropouts to 0
    5. Normalization layers probably should stay untouched.
    """

    def turn_on_model_regularizations(self):
        if not self.regularized:
            for param in self.parameters():
                name, obj = param
                if isinstance(obj, torch.nn.Dropout):
                    obj.p = self.unregularized_params[name]

            self.configure_optimizers = self.original_configure_optimizers

            self.regularized = True

    def turn_off_model_reguralizations(self):
        if self.regularized:
            self.unregularized_params = {}
            for param in self.parameters():
                name, obj = param
                if isinstance(obj, torch.nn.Dropout):
                    self.unregularized_params[name] = obj.p
                    obj.p = 0

            # Then make it returning just simple Adam, no fancy optimizers at this stage
            self.original_configure_optimizers = self.configure_optimizers
            self.configure_optimizers = lambda self: torch.optim.Adam(
                self.parameters(), lr=3e-4
            )

            self.regularized = False
