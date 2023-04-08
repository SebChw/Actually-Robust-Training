from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


class SingleThresholdLabelStrategy():
    def __init__(self):
        self.threshold = float("Inf")
        self.all_losses = []

    def __call__(self, losses):
        return losses*(losses <= self.threshold)

    def update(self, losses: Dict[str, Dict[str, np.ndarray]]):
        self.all_losses = np.concatenate([losses[song][instrument] for song in losses for instrument in losses[song]]).flatten()
        self.old_threshold = self.threshold
        self.threshold = np.percentile(self.all_losses, 90)

    def get_metrics(self):
        return {
            "label_threshold": self.threshold
        }

    def get_figures(self):
        fig = plt.figure(num=1, clear=True)
        ax = fig.gca()
        ax.hist(self.all_losses, bins=50)
        ax.set_title("L1 losses histogram")
        ax.axvline(x = min(self.old_threshold, max(self.all_losses)), color = 'r', label = 'old threshold')
        ax.axvline(x=min(self.threshold, max(self.all_losses)), color='g', label='threshold')
        return {
            "histogram": fig
        }
