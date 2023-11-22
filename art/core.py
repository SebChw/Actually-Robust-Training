import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import lightning as L
import torch.nn
from torch.utils.data import DataLoader

from art.metrics import MetricCalculator
from art.utils.enums import LOSS, PREDICTION, TARGET


class ArtModule(L.LightningModule, ABC):
    def __init__(
        self,
    ):
        super().__init__()
        self.regularized = True
        self.reset_pipelines()

    """
    A module for managing the training process and application of various model configurations.
    """

    def set_metric_calculator(self, metric_calculator: MetricCalculator):
        """
        Set metric calculator.

        Args:
            metric_calculator (MetricCalculator): A metric calculator.
        """
        self.metric_calculator = metric_calculator

    def check_setup(self):
        """
        Check if the metric calculator has been set.

        Raises:
            ValueError: If the metric calculator has not been set.
        """
        if not hasattr(self, "metric_calculator"):
            raise ValueError("You need to set metric calculator first!")

    def reset_pipelines(self):
        """
        Reset pipelines for training, validation, and testing.
        """
        self.train_step_pipeline = [
            self.parse_data,
            self.predict,
            self.compute_metrics,
            self.compute_loss,
        ]
        self.validation_step_pipeline = [
            self.parse_data,
            self.predict,
            self.compute_metrics,
            self.compute_loss,
        ]
        self.ml_train_pipeline = [self.ml_parse_data, self.baseline_train]

    def turn_on_model_regularizations(self):
        """
        Turn on model regularizations.
        """
        if not self.regularized:
            for param in self.parameters():
                name, obj = param
                if isinstance(obj, torch.nn.Dropout):
                    obj.p = self.unregularized_params[name]

            self.configure_optimizers = self.original_configure_optimizers

            self.regularized = True

    def turn_off_model_reguralizations(self):
        """
        Turn off model regularizations.
        """
        if self.regularized:
            self.unregularized_params = {}
            for param in self.parameters():
                name, obj = param
                if isinstance(obj, torch.nn.Dropout):
                    self.unregularized_params[name] = obj.p
                    obj.p = 0

            # Simple Adam, no fancy optimizers at this stage
            self.original_configure_optimizers = self.configure_optimizers
            self.configure_optimizers = lambda self: torch.optim.Adam(
                self.parameters(), lr=3e-4
            )

            self.regularized = False

    def parse_data(self, data: Dict):
        """
        Parse data.

        Args:
            data (Dict): Data to parse.

        Returns:
            Dict: Parsed data.
        """
        return data

    def predict(self, data: Dict):
        """
        Predict.

        Args:
            data (Dict): Data to predict.

        Returns:
            Dict: Data with predictions.
        """
        return data

    def compute_loss(self, data: Dict):
        """
        Compute loss.

        Args:
            data (Dict): Data to compute loss.

        Returns:
            Dict: Data with loss.
        """
        return data

    def compute_metrics(self, data: Dict):
        """
        Compute metrics.

        Args:
            data (Dict): Data to compute metrics.

        Returns:
            Dict: Data with metrics.
        """
        self.metric_calculator(self, data)
        return data

    def validation_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        """
        Validation step.

        Args:
            batch (Union[Dict[str, Any], DataLoader, torch.Tensor]): Batch to validate.
            batch_idx (int): Batch index.
        """
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.validation_step_pipeline:
            data = func(data)

    def training_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        """
        Training step.

        Args:
            batch (Union[Dict[str, Any], DataLoader, torch.Tensor]): Batch to train.
            batch_idx (int): Batch index.

        Returns:
            Dict: Data with loss.
        """
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.train_step_pipeline:
            data = func(data)

        return data[LOSS]

    def test_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        """
        Test step.

        Args:
            batch (Union[Dict[str, Any], DataLoader, torch.Tensor]): Batch to test.
            batch_idx (int): Batch index.
        """
        data = {"batch": batch, "batch_idx": batch_idx}
        for func in self.validation_step_pipeline:
            data = func(data)

    def ml_parse_data(self, data: Dict):
        """
        Parse data for machine learning training.

        Args:
            data (Dict): Data to parse.

        Returns:
            Dict: Parsed data.
        """
        return data

    def baseline_train(self, data: Dict):
        """
        Baseline train.

        Args:
            data (Dict): Data to train.

        Returns:
            Dict: Data with loss.
        """
        return data

    def ml_train(self, data: Dict):
        """
        Machine learning train.

        Args:
            data (Dict): Data to train.

        Returns:
            Dict: Data with loss.
        """
        for func in self.ml_train_pipeline:
            data = func(data)

        return data

    def unify_type(self: Any, x: Any):
        """
        Unify type - x to torch.Tensor.

        Args:
            x (Any): Data to unify type.

        Returns:
            torch.Tensor: Data with unified type."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        return x

    def prepare_for_metric(self: Any, data: Dict):
        """
        Prepare data for metric calculation.

        Args:
            data (Dict): Data to prepare.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data with unified type.
        """
        preds, targets = data[PREDICTION], data[TARGET]

        return self.unify_type(preds), self.unify_type(targets)

    @abstractmethod
    def log_params(self):
        pass
