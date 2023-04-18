import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from art.enums import TrainingStage
from art.similarity.networks import EfficientWordNet
from art.utils.metrics import get_metric_array


class EfficientWordNetOriginal(L.LightningModule):
    def __init__(
        self,
        network=EfficientWordNet(),
        distance_func=nn.PairwiseDistance(p=2, keepdim=True),
    ):
        super().__init__()
        self.network = network
        self.distance_func = distance_func

        self.accuracies = get_metric_array(Accuracy, task="binary")

    def forward(self, X):
        x1, x2 = X["X1"], X["X2"]
        return self.distance_func(self.network(x1), self.network(x2))

    def _weird_loss(self, y_true, y_pred):
        # They define some weird kind of BCE loss based on distances
        # Here they opt definitelly for Euclidean distance
        match_loss = y_true * -2.0 * torch.log(1 - y_pred / 2)
        mismatch_loss = torch.maximum(
            (1 - y_true) * (-torch.log(y_pred / 0.2)), torch.zeros_like(y_true)
        )

        return torch.mean(match_loss + mismatch_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _threshold_func(self, dist_pred):
        # They measure accuracy assuming threshold equal to 0.2
        return dist_pred <= 0.2

    def processing_step(self, batch, stage: TrainingStage):
        # y=1 -> examples are from the same class
        # y=0 -> examples are from different classes
        #! Some inputs seems to be nans!
        X1, X2, y = batch["X1"], batch["X2"], batch["y"]

        emb1, emb2 = self.network(X1), self.network(X2)
        dist_pred = self.distance_func(emb1, emb2)

        loss = self._weird_loss(y, dist_pred)

        self.log(f"{stage.name}_loss", loss, prog_bar=True)

        y_pred = self._threshold_func(dist_pred)
        stage_accuracy = self.accuracies[stage.value]
        stage_accuracy(y_pred, y)
        self.log(f"{stage.name}_acc", stage_accuracy, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, TrainingStage.TRAIN)

    def validation_step(self, batch, batch_idx):
        return self.processing_step(batch, TrainingStage.VALIDATION)

    def test_step(self, batch, batch_idx):
        return self.processing_step(batch, TrainingStage.TEST)
