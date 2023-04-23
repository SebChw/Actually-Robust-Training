import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import numpy as np
from collections import defaultdict
from art.utils.plotters import SourceSepPlotter
from art.utils.sourcesep_augment import Scale, Shift, FlipSign, FlipChannels, Remix


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


class LitAudioSourceSeparator(L.LightningModule):
    def __init__(
        self,
        model,
        sources=["bass", "vocals", "drums", "other"],
        calculate_sdr=False,
        wrong_label_strategy=None,
        plotter=SourceSepPlotter(),
        augment=True,
    ):
        super().__init__()
        self.sources = sources
        self.model = model
        self.calculate_sdr = calculate_sdr
        self.wrong_label_strategy = wrong_label_strategy
        self.plotter = plotter
        if augment:
            self.transform = nn.Sequential(
            Shift(),
            FlipChannels(),
            FlipSign(),
            Scale(),
            Remix(),
            )
        else:
            self.transform = nn.Identity()

        if calculate_sdr:
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

    def _update_song_losses(self, prompt, batch, loss):
        song_names = batch["name"]
        numbers_of_window = batch["n_window"]
        for song_id, (song_name, num_of_win) in enumerate(
            zip(song_names, numbers_of_window)
        ):
            for instrument_id, instrument in enumerate(self.sources):
                self.song_losses[prompt + "_" + song_name][instrument][
                    num_of_win
                ] = loss[song_id][instrument_id]

    def processing_step(self, batch, prompt):
        X = batch["mixture"]
        target = batch["target"]

        predictions = self.model(X)

        loss = F.l1_loss(predictions, target, reduction="none").mean(dim=(-1, -2))

        # At this point loss has shape (n_songs, n_instruments)
        self._update_song_losses(prompt, batch, loss)
        if self.wrong_label_strategy and prompt == "train":
            loss = self.wrong_label_strategy(loss)
        loss = loss.mean()

        self.log(
            f"{prompt}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=X.shape[0],
        )

        if self.calculate_sdr:
            try:
                for i, (source, sdr) in enumerate(self.sdr.items()):
                    sdr(predictions[:, i, ...], target[:, i, ...])
                    self.log(
                        f"{prompt}_{source}_sdr",
                        sdr,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                    )
            except np.linalg.LinAlgError:
                print("SINGULARITY IN SDR!")

        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.processing_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.processing_step(batch, "test")

    def on_train_epoch_end(self):
        if self.wrong_label_strategy:
            self.wrong_label_strategy.update(self.song_losses["train"])
            self.logger.log_metrics(self.wrong_label_strategy.get_metrics())
            for key, fig in self.wrong_label_strategy.get_figures().items():
                self.logger.experiment[
                    f"loss_thresholds/epoch{self.current_epoch}/{key}"
                ].upload(fig)

    def on_validation_epoch_end(self):
        self.plotter.update(self)

    def on_fit_start(self):
        self.on_train_epoch_start()

    def on_train_epoch_start(self):
        self.song_losses = defaultdict(
            lambda: defaultdict(
                lambda: {source: np.zeros(100) for source in self.sources}
            )
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        x, y = batch["mixture"], batch["target"]
        batch["mixture"] = self.transform(x)
        return x, y
