import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import numpy as np


class LitAudioClassifier(L.LightningModule):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def processing_step(self, batch, prompt):
        x, y = batch["data"], batch["label"]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log(f"{prompt}_loss", loss)

        self.accuracy(logits, y)
        self.log(f"{prompt}_acc", self.accuracy, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.processing_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.processing_step(batch, "test")


class LitAudioSourceSeparator(L.LightningModule):
    def __init__(self, model, sources=["bass", "vocals", "drums", "other"]):
        super().__init__()

        self.sources = sources
        self.model = model
        # !one may use MetricCollection wrapper but not in this case
        self.sdr = nn.ModuleDict(
            {source: torchmetrics.SignalDistortionRatio() for source in sources}
        )

    def forward(self, X):
        # Here we can make it more like inference step and return dict with sources
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def processing_step(self, batch, prompt):
        X = batch["mixture"]
        target = batch["target"]

        predictions = self.model(X)

        loss = F.l1_loss(predictions, target)
        self.log(f"{prompt}_loss", loss)

        try:
            # !If some target is entirely 0 then this sdr calculation fails :( flattening could help but then I get memory errors
            for i, (source, sdr) in enumerate(self.sdr.items()):
                sdr(predictions[:, i, ...], target[:, i, ...])
                self.log(f"{prompt}_{source}_sdr", sdr, on_step=True, on_epoch=True)
        except np.linalg.LinAlgError:
            print("SINGULARITY IN SDR!")

        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.processing_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.processing_step(batch, "test")
