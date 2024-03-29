import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import lightning as L
import torch.nn
from torch.utils.data import DataLoader

from art.metrics import MetricCalculator
from art.utils.enums import LOSS, PREDICTION, TARGET, TrainingStage


class ArtModule(L.LightningModule, ABC):
    def __init__(
        self,
    ):
        super().__init__()
        self.regularized = True
        self.set_pipelines()
        self.stage: TrainingStage = TrainingStage.TRAIN

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

    def set_pipelines(self):
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
        self.stage = TrainingStage.VALIDATION
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
        self.stage = TrainingStage.TRAIN
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
        self.stage = TrainingStage.TEST
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
