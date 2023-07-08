import random

import numpy as np
import torch
import torch.nn as nn


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
