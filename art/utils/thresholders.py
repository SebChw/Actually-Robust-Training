from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")


class SingleThresholdLabelStrategy:
    def __init__(self, percentile=80, sources=[]):
        self.thresholds = float("Inf")
        self.all_losses = []
        self.percentile = percentile
        self.sources = sources

    def __call__(self, losses):
        return losses * (losses <= self.thresholds)

    def update(self, losses: Dict[str, Dict[str, np.ndarray]]):
        self.all_losses = np.concatenate(
            [
                np.trim_zeros(losses[song][instrument], "b")
                for song in losses
                for instrument in losses[song]
            ]
        ).flatten()
        self.old_threshold = self.thresholds
        self.thresholds = np.percentile(self.all_losses, self.percentile)

    def get_metrics(self):
        return {"label_threshold": self.thresholds}

    def get_figures(self):
        hist = self.make_threshold_histogram(self.all_losses, self.old_threshold, self.thresholds)
        return {
            "histogram": hist
        }

    def make_threshold_histogram(self, losses, old_threshold, threshold, title="threshold"):
        shifted_losses = losses + 1  # for log scale
        fig = plt.figure(num=1, clear=True)
        ax = fig.gca()
        max_all_losses = max(shifted_losses)
        ax.hist(shifted_losses, bins=np.logspace(np.log10(1), np.log10(max_all_losses), 50))
        ax.set_title(f"1 + L1 losses {title}")
        ax.axvline(x=min(old_threshold + 1, max(shifted_losses)), color='r', label='old threshold')
        ax.axvline(x=min(threshold + 1, max(shifted_losses)), color='g', label='threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')

        return fig


class InstrumentThresholdLabelStrategy(SingleThresholdLabelStrategy):
    def __init__(self, percentile = 95, sources = [], device = "cuda:0"):
        self.thresholds = torch.Tensor([float("Inf")]* len(sources)).to(device)
        self.old_thresholds = torch.Tensor([float("Inf")]* len(sources)).to(device)
        self.all_losses = []
        self.percentile = percentile
        self.sources = sources
        self.sources_losses = {source:[] for source in sources}
        self.device = device

    def update(self, losses: Dict[str, Dict[str, np.ndarray]]):
        self.sources_losses = {source: [] for source in self.sources}
        for song in losses:
            for source in losses[song]:
                self.sources_losses[source].append(np.trim_zeros(losses[song][source]))
        self.thresholds = []
        for source in self.sources:
            self.sources_losses[source] = np.concatenate(self.sources_losses[source]).flatten()
            self.thresholds.append(np.percentile(self.sources_losses[source], self.percentile))
        self.thresholds = torch.Tensor(self.thresholds).to(self.device)

    def get_metrics(self):
        return {
            f"{instrument}_label_threshold": threshold for instrument, threshold in zip(self.sources, self.thresholds)
        }

    def get_figures(self):
        figures = {}
        for i, source in enumerate(self.sources):
            figures[f"{source}_histogram"] = self.make_threshold_histogram(self.sources_losses[source], self.old_thresholds[i].cpu(), self.thresholds[i].cpu(), title=source)
        return figures

