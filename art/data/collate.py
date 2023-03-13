import random
import torch.nn as nn
import torch
import numpy as np
from collections import defaultdict

################################################################################
################# The idea is to have a factory of collate functions#############
################################################################################
#! Or maybe use HF DataCollators? and move entireley to HF?


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
            x, label = item['audio']['array'], item['label']

            if x.shape[0] > max_length:
                random_offset = random.randint(0, x.shape[0] - max_length)
                x = x[random_offset: random_offset + max_length]

            if normalize:
                (x - normalize["mean"]) / np.sqrt(normalize["var"] + 1e-7)

            X.append(x)
            labels.append(label)

        labels = torch.tensor(labels)
        X = nn.utils.rnn.pad_sequence(X, batch_first=True)

        return {"data": X.unsqueeze(1), "label": labels}

    return waveform_collate_fn


def create_sourceseparation_collate(normalize=False, length=44100, instruments=["bass", "vocals", "drums", "other"]):
    def waveform_collate_fn(batch):
        X = defaultdict(lambda: [])

        for item in batch:
            instruments_wavs = {name: item[name]
                                ["array"] for name in instruments}

            random_offset = random.randint(
                0, instruments_wavs[instruments[0]].shape[1] - length)

            instruments_wavs = {name: wav[:, random_offset: random_offset + length]
                                for name, wav in instruments_wavs.items()}

            instruments_wavs["mixture"] = torch.stack(
                list(instruments_wavs.values()), 0).sum(axis=0)

            for name, waveform in instruments_wavs.items():
                X[name].append(waveform)

        separations = [torch.stack(X[instrument])
                       for instrument in instruments]

        return {"mixture": torch.stack(X["mixture"]), "target": torch.stack(separations, dim=1)}

    return waveform_collate_fn
