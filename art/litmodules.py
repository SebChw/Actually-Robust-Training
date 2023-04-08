import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


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


class WrongLabelStrategy:
    def __init__(self, strategy="threshold", **strategy_kwargs):
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs

    def __call__(self, losses):
        # TODO
        pass


class LitAudioSourceSeparator(L.LightningModule):
    def __init__(
        self,
        model,
        sources=["bass", "vocals", "drums", "other"],
        calculate_sdr=False,
        wrong_label_strategy=None,
    ):
        super().__init__()

        self.sources = sources
        self.model = model
        self.calculate_sdr = calculate_sdr
        self.wrong_label_strategy = wrong_label_strategy

        # For every song and every instrument we track losses
        # Ill trim these zeros later on
        self.song_losses = defaultdict(
            lambda: {source: np.zeros(100) for source in self.sources}
        )

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
                self.song_losses[prompt + song_name][instrument][num_of_win] = loss[
                    song_id
                ][instrument_id]

    def processing_step(self, batch, prompt):
        X = batch["mixture"]
        target = batch["target"]

        predictions = self.model(X)

        loss = F.l1_loss(predictions, target, reduction="none").mean(dim=(-1, -2))

        #! At this point loss has shape (n_songs, n_instruments)
        self._update_song_losses(prompt, batch, loss)
        if self.wrong_label_strategy and prompt == "train":
            loss = self.wrong_label_strategy(loss)

        loss = loss.mean()

        self.log(f"{prompt}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

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
        self.processing_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.processing_step(batch, "test")

    def on_validation_epoch_end(self):
        # Saving all necessary plots to the logger
        for song, instrument_losses in self.song_losses.items():
            fig, ax = plt.subplots(1, 4, figsize=(30, 10))
            title = f"Epoch_{self.current_epoch}_song_{song}"
            fig.suptitle(title, fontsize=16)
            for i, (instrument, loss) in enumerate(instrument_losses.items()):
                loss = np.trim_zeros(loss, "b")
                # ax[i].bar(np.arange(len(loss)), loss)
                ax[i].plot(loss, "bo-")
                ax[i].set_title(instrument)
                ax[i].set_xlabel("number of window")
                ax[i].set_ylabel("L1")

            self.logger.experiment[title].upload(fig)

        self.song_losses = defaultdict(
            lambda: {source: np.zeros(100) for source in self.sources}
        )
