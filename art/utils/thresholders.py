from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")


class SingleThresholdLabelStrategy():
    def __init__(self, percentile = 80):
        self.threshold = float("Inf")
        self.all_losses = []
        self.percentile = percentile

    def __call__(self, losses):
        return losses * (losses <= self.threshold)

    def update(self, losses: Dict[str, Dict[str, np.ndarray]]):
        self.all_losses = np.concatenate(
            [
                np.trim_zeros(losses[song][instrument], "b")
                for song in losses
                for instrument in losses[song]
            ]
        ).flatten()
        self.old_threshold = self.threshold
        self.threshold = np.percentile(self.all_losses, self.percentile)

    def get_metrics(self):
        return {"label_threshold": self.threshold}

    def get_figures(self):
        shifted_losses = self.all_losses +1#for log scale
        fig = plt.figure(num=1, clear=True)
        ax = fig.gca()
        max_all_losses = max(shifted_losses)
        ax.hist(shifted_losses, bins=np.logspace(np.log10(1), np.log10(max_all_losses), 50))
        ax.set_title("1 + L1 losses histogram")
        ax.axvline(x = min(self.old_threshold+1, max(shifted_losses)), color = 'r', label = 'old threshold')
        ax.axvline(x=min(self.threshold+1, max(shifted_losses)), color='g', label='threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        return {
            "histogram": fig
        }
