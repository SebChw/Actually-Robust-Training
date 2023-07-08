import random
from collections import defaultdict

import torch

from art.utils.sourcesep_augment import FlipChannels, FlipSign, Remix, Scale


class SourceSeparationCollate:
    def __init__(
        self,
        max_length=-1,
        instruments=["bass", "vocals", "drums", "other"],
        augment=False,
    ):
        self.max_length = max_length
        self.instruments = instruments
        if augment:
            self.augments = [
                Scale(),
                # Shift(), - need to add some padding to use it.
                FlipSign(),
                FlipChannels(),
                Remix(),
            ]
        else:
            self.augments = []

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
        for augment in self.augments:
            # choose random 8% samples from separations tensor
            n_samples = int(separations.shape[0] * 0.08)
            rand_idx = torch.randperm(separations.shape[0])[:n_samples]
            random_samples = separations[rand_idx]
            separations = torch.cat([separations, augment(random_samples)], dim=0)

        separations = (
            separations - torch.tensor(means).to(separations)[:, None, None, None]
        ) / torch.tensor(stds).to(separations)[:, None, None, None]

        return {
            "mixture": torch.sum(separations, axis=1),
            "target": separations,
            "name": song_names,
            "n_window": n_windows,
        }
