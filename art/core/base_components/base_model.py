import hashlib
import inspect
from typing import Any, Dict, Union

import lightning as L
import torch.nn
from torch.utils.data import DataLoader

from art.core.MetricCalculator import MetricCalculator
from art.utils.enums import LOSS, PREDICTION, TARGET

from abc import ABC, abstractmethod


class ArtModule(L.LightningModule, ABC):
    def __init__(
        self,
    ):
        super().__init__()
        self.regularized = True
        self.reset_pipelines()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    I think in case of models we can easily write some general purpose function that will do the job.

    But maybe this should be specific to the Stage we are currently at?

    1. Turn of weight decay if turned on -> easy to do in optimizer.
    2. Don't use fancy optimizers -> stick to Adam (But maybe this depends on stage?)
    3. Turn off learning rate decays -> Again, maybe this depends on stage?
    4. Set all dropouts to 0
    5. Normalization layers probably should stay untouched.
    """

    @abstractmethod
    def log(self, metric_name: str, metric_val: str, on_step: Any = False, on_epoch: Any = True):
        pass

    def set_metric_calculator(self, metric_calculator: MetricCalculator):
        self.metric_calculator = metric_calculator

    def check_setup(self):
        if not hasattr(self, "metric_calculator"):
            raise ValueError("You need to set metric calculator first!")

    def reset_pipelines(self):
        # THIS FUNCTION IS NECESSARY AS WE MAY DECORATE AND DECORATED FUNCTIONS WON'T BE USED!
        # This probably should be splitted into 2 classes. One for model another for baseline.
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

    def parse_data(self, data: Dict):
        return data

    def predict(self, data: Dict):
        return data

    # I wonder if compute_loss could be somehow passed to compute_metrics.
    # But I see some limitations.
    def compute_loss(self, data: Dict):
        return data

    def compute_metrics(self, data: Dict):
        self.metric_calculator(self, data)
        return data

    def validation_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.validation_step_pipeline:
            data = func(data)

    def training_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.train_step_pipeline:
            data = func(data)

        return data[LOSS]

    def test_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.validation_step_pipeline:
            data = func(data)

    def ml_parse_data(self, data: Dict):
        return data

    def baseline_train(self, data: Dict):
        return data

    def ml_train(self, data: Dict):
        for func in self.ml_train_pipeline:
            data = func(data)

        return data

    def get_hash(self):
        return hashlib.md5(
            inspect.getsource(self.__class__).encode("utf-8")
        ).hexdigest()

    def unify_type(self: Any, x: Any):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        return x

    def prepare_for_metric(self: Any, data: Dict):
        preds, targets = data[PREDICTION], data[TARGET]

        return self.unify_type(preds), self.unify_type(targets)
