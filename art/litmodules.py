import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LitAudioClassifier(L.LightningModule):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model

        # Why this is necessary in common pitfalls
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        self.accuracy = nn.ModuleList(
            [
                torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                for _ in range(3)
            ]
        )
        self.str_to_stage = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def processing_step(self, batch, stage):
        x, y = batch["data"], batch["label"]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        stage_id = self.str_to_stage[stage]
        self.accuracy[stage_id](logits, y)
        self.log(
            f"{stage}_acc",
            self.accuracy[stage_id],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.processing_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.processing_step(batch, "test")
