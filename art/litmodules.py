from collections import defaultdict

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from art.utils.plotters import SourceSepPlotter


class LitAudioSourceSeparator(L.LightningModule):
    def __init__(
        self,
        model,
        sources=["bass", "vocals", "drums", "other"],
        calculate_sdr=False,
        wrong_label_strategy=None,
        plotter=SourceSepPlotter(),
        warmup_epochs=5,
    ):
        super().__init__()
        self.sources = sources
        self.model = model
        self.calculate_sdr = calculate_sdr
        self.wrong_label_strategy = wrong_label_strategy
        self.plotter = plotter
        self.warmup_epochs = warmup_epochs
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
                self.song_losses[prompt][song_name][instrument][num_of_win] = loss[
                    song_id
                ][instrument_id]

    def processing_step(self, batch, prompt):
        X = batch["mixture"]
        target = batch["target"]

        predictions = self.model(X)

        loss = F.l1_loss(predictions, target, reduction="none").mean(dim=(-1, -2))

        # At this point loss has shape (n_songs, n_instruments)
        self._update_song_losses(prompt, batch, loss)
        if (
            self.current_epoch > self.warmup_epochs
            and self.wrong_label_strategy
            and prompt == "train"
        ):
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
        if self.current_epoch > self.warmup_epochs and self.wrong_label_strategy:
            self.wrong_label_strategy.update(self.song_losses["train"])
            self.logger.log_metrics(self.wrong_label_strategy.get_metrics())
            for key, fig in self.wrong_label_strategy.get_figures().items():
                self.logger.experiment[
                    f"loss_thresholds/epoch{self.current_epoch}_{key}"
                ].upload(fig)

    def on_validation_epoch_end(self):
        self.plotter.update(self)

    def on_fit_start(self):
        self.on_train_epoch_start()

    def on_train_epoch_start(self):
        # init with -1 to filter out later
        self.song_losses = defaultdict(
            lambda: defaultdict(
                lambda: {source: np.full(100, -1) for source in self.sources}
            )
        )
