import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from art.enums import TrainingStage
from art.similarity.efficientwordnet.networks import EfficientWordNet
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

        self.N_BINS = 10
        self.RANGE = (0, 2)
        self.histograms = {}
        self.bins = np.histogram([], self.N_BINS, range=self.RANGE, density=False)[1]
        self.reset_histograms(TrainingStage.TRAIN)
        self.reset_histograms(TrainingStage.VALIDATION)

    def forward(self, X):
        x1, x2 = X["X1"], X["X2"]
        return self.distance_func(self.network(x1), self.network(x2))

    def _weird_loss(self, y_true, y_pred):
        # They define some weird kind of BCE loss based on distances
        # Here they opt definitelly for Euclidean distance

        # y_pred -> distance between 2 embeddings. Vectors are normalized so distance is at most 2
        # y_true -> 1 if they are similar, 0 if they are not
        match_loss = y_true * -2.0 * torch.log(1 - y_pred / 2)  # y_pred (0, 2)
        mismatch_loss = torch.maximum(
            (1 - y_true) * (-torch.log(y_pred / 0.2)),
            torch.zeros_like(
                y_true
            ),  #! 0.2 should be rather 2, if it is 0.2 and we have log(10) we get big negative loss.
        )

        return torch.mean(match_loss + mismatch_loss)

    # def different_loss(self, euclidean_distance, label):
    #     MARGIN = 0.2
    #     pos = (1 - label) * torch.pow(euclidean_distance, 2)
    #     neg = (label) * torch.pow(torch.clamp(MARGIN - euclidean_distance, min=0.0), 2)
    #     loss_contrastive = torch.mean(pos + neg)
    #     return loss_contrastive

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer

    def _threshold_func(self, dist_pred):
        # They measure accuracy assuming threshold equal to 0.2
        return dist_pred <= 0.2

    def reset_histograms(self, stage):
        self.histograms[stage] = {
            "positive": np.histogram([], self.N_BINS, range=self.RANGE, density=False)[
                0
            ],
            "negative": np.histogram([], self.N_BINS, range=self.RANGE, density=False)[
                0
            ],
        }

    def update_histograms(self, dist_pred, y, stage):
        dist_pred = dist_pred.detach().cpu().numpy().flatten()
        y = y.detach().cpu().numpy().flatten()
        self.histograms[stage]["positive"] += np.histogram(
            dist_pred[y == 1], self.N_BINS, range=self.RANGE, density=False
        )[0]
        self.histograms[stage]["negative"] += np.histogram(
            dist_pred[y == 0], self.N_BINS, range=self.RANGE, density=False
        )[0]

    def plot_histograms(self, stage):
        fig, ax = plt.subplots(1, 2, num=1, clear=True)
        fig.suptitle(f"distribution of distances between embedding during {stage}")
        ax[0].bar(self.bins[:-1], self.histograms[stage]["positive"], width=0.1)
        ax[0].set_title("positive examples")
        ax[1].bar(self.bins[:-1], self.histograms[stage]["negative"], width=0.1)
        ax[1].set_title("negative examples")

        fig.savefig(f"hist_{stage}_{self.current_epoch}.png")

    def processing_step(self, batch, stage: TrainingStage):
        # y=1 -> examples are from the same class
        # y=0 -> examples are from different classes
        X1, X2, y = batch["X1"], batch["X2"], batch["y"]
        batch_size = X1.shape[0]
        X = torch.cat([X1, X2], dim=0)
        m, s = X.mean(dim=(1, 2, 3), keepdim=True), X.std(dim=(1, 2, 3), keepdim=True)
        embeddings = self.network((X - m) / s)
        emb1, emb2 = embeddings[:batch_size], embeddings[batch_size:]

        # Here I can easily gather information about distances and make distribution out of them.
        dist_pred = self.distance_func(emb1, emb2)

        self.update_histograms(dist_pred, y, stage)

        loss = self._weird_loss(y, dist_pred)
        # loss = self.different_loss(dist_pred, y)

        self.log(f"{stage.name}_loss", loss, prog_bar=True, batch_size=batch_size)

        y_pred = self._threshold_func(dist_pred)
        stage_accuracy = self.accuracies[stage.value]
        stage_accuracy(y_pred, y)
        self.log(
            f"{stage.name}_acc", stage_accuracy, prog_bar=True, batch_size=batch_size
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, TrainingStage.TRAIN)

    def validation_step(self, batch, batch_idx):
        return self.processing_step(batch, TrainingStage.VALIDATION)

    def test_step(self, batch, batch_idx):
        return self.processing_step(batch, TrainingStage.TEST)

    def on_train_epoch_end(self):
        self.plot_histograms(TrainingStage.TRAIN)
        self.reset_histograms(TrainingStage.TRAIN)

    def on_validation_epoch_end(self):
        self.plot_histograms(TrainingStage.VALIDATION)
        self.reset_histograms(TrainingStage.VALIDATION)


class NewSota(L.LightningModule):
    """Implement your own sota model here."""

    pass