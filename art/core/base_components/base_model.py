from typing import Any, Optional

import lightning.pytorch as pl
import torch.nn

from art.enums import LOSS, TrainingStage
from art.metric_calculator import MetricCalculator


class ArtModule(pl.LightningModule):
    def __init__(
        self,
        # device="cpu",
    ):
        super().__init__()
        self.regularized = True
        # print(self.device)
        self.metric_calculator = MetricCalculator()
        self.reset_pipelines()

    """
    I think in case of models we can easily write some general purpose function that will do the job.

    But maybe this should be specific to the Stage we are currently at?

    1. Turn of weight decay if turned on -> easy to do in optimizer.
    2. Don't use fancy optimizers -> stick to Adam (But maybe this depends on stage?)
    3. Turn off learning rate decays -> Again, maybe this depends on stage?
    4. Set all dropouts to 0
    5. Normalization layers probably should stay untouched.
    """

    def reset_pipelines(self):
        # THIS FUNCTION IS NECESSARY AS WE MAY DECORATE AND DECORATED FUNCTIONS WON'T BE USED!
        #! This probably should be splitted into 2 classes. One for model another for baseline.
        self.train_step_pipeline = [
            self.parse_data,
            self.predict,
            self.compute_metrics,
            self.compute_loss,
        ]  # STRATEGY DESIGN PATTERN
        self.validation_step_pipeline = [
            self.parse_data,
            self.predict,
            self.compute_metrics,
            self.compute_loss,
        ]
        self.ml_train_pipeline = [self.ml_parse_data, self.baseline_train]

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

    def parse_data(self, data):
        return data

    def predict(self, data):
        return data

    # I wonder if compute_loss could be somehow passed to compute_metrics.
    # But I see some limitations.
    def compute_loss(self, data):
        return data

    def compute_metrics(self, data):
        self.metric_calculator(self, data)
        return data

    def validation_step(self, batch, batch_idx):
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.validation_step_pipeline:
            data = func(data)

    def training_step(self, batch, batch_idx):
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.train_step_pipeline:
            data = func(data)

        return data[LOSS]

    def ml_parse_data(self, data):
        return data

    def baseline_train(self, data):
        return data

    def ml_train(self, data):
        for func in self.ml_train_pipeline:
            data = func(data)

        return data
