import random
import torch.nn as nn
import torch
import numpy as np
from collections import defaultdict


def create_waveform_collate(normalize=None, max_length=16000):
    """Use this collator factory if you don't want to use feature_extractor that will create
    cache file 2x bigger than original dataset due to float32 instead of float16
    Args:
        normalize (_type_, optional): _description_. Defaults to None.
        max_length (int, optional): _description_. Defaults to 16000.
    """

    def waveform_collate_fn(batch):
        X, labels = [], []
        for item in batch:
            x, label = item["audio"]["array"], item["label"]

            if x.shape[0] > max_length:
                random_offset = random.randint(0, x.shape[0] - max_length)
                x = x[random_offset : random_offset + max_length]

            if normalize:
                (x - normalize["mean"]) / np.sqrt(normalize["var"] + 1e-7)

            X.append(x)
            labels.append(label)

        labels = torch.tensor(labels)
        X = nn.utils.rnn.pad_sequence(X, batch_first=True)

        return {"data": X.unsqueeze(1), "label": labels}

    return waveform_collate_fn


class SourceSeparationCollate:
    def __init__(self, max_length=-1, instruments=["bass", "vocals", "drums", "other"]):
        self.max_length = max_length
        self.instruments = instruments

    def __call__(self, batch):
        X = defaultdict(lambda: [])
        means = []
        stds = []
        song_names = []
        n_windows = []
        for item in batch:
            song_names.append(item["name"].split("/")[-1])
            n_windows.append(item["n_window"])

            means.append(item["mean"])
            stds.append(item["std"])
            instruments_wavs = {name: item[name]["array"] for name in self.instruments}

            if self.max_length != -1:
                random_offset = random.randint(
                    0, instruments_wavs[self.instruments[0]].shape[1] - self.max_length
                )

                instruments_wavs = {
                    name: wav[:, random_offset : random_offset + self.max_length]
                    for name, wav in instruments_wavs.items()
                }

            for name, waveform in instruments_wavs.items():
                X[name].append(waveform)

        separations = [torch.stack(X[instrument]) for instrument in self.instruments]
        separations = torch.stack(separations, axis=1)

        separations = (
            separations - torch.tensor(means).to(separations)[:, None, None, None]
        ) / torch.tensor(stds).to(separations)[:, None, None, None]

        return {
            "mixture": torch.sum(separations, axis=1),
            "target": separations,
            "name": song_names,
            "n_window": n_windows,
        }
