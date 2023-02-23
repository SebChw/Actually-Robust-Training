from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch.nn as nn
import torch

import datasets

from torch.utils.data.sampler import BatchSampler, RandomSampler
import random
import numpy as np


def dummy_classification_generator(n_samples=64, shape=(1, 16000), label=0):
    for i in range(n_samples):
        yield {"data": torch.randn(shape), "label": label}


def create_waveform_collate(normalize=None, max_length=16000):
    #!Operating on audio directly instead of using predefined feature extractor in case
    #!Of raw waveform is much more efficient as hf doesn't support float16
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


class SanityCheckDataModule(pl.LightningDataModule):
    def __init__(self, dataset_generator):
        super().__init__()
        self.dataset_generator = dataset_generator

    def prepare_data(self):
        #!If you don't write anything here Lightning will try to use prepare_data_per_node
        print("Nothing to download")

    def setup(self, stage):
        self.dataset = datasets.Dataset.from_generator(self.dataset_generator)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.dataset.with_format("torch", device=device)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=32)


class GoogleCommandDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, spectrogram=False):
        super().__init__()

        self.batch_size = batch_size
        self.spectrogram = spectrogram
        if self.spectrogram:
            self.collate = lambda x: x
        else:
            self.collate = create_waveform_collate()
        self.dataset = None

    def prepare_data(self):
        #!If you don't write anything here Lightning will try to use prepare_data_per_node
        print("Nothing to download")

    def setup(self, stage):
        # This separation between assignment and donwloading may be usefull for multi-node training
        if self.dataset is None:
            self.dataset = datasets.load_dataset("speech_commands", 'v0.02')
            self.dataset = self.dataset.remove_columns(
                ["file", "is_unknown", "speaker_id", "utterance_id"])

            #! This is very inefficient it stores stuff on 64 bits by default. HF doesn't support 16 bits.
            # self.dataset = self.dataset.map(
            #     lambda example: {"data": np.expand_dims(example["audio"]["array"], 0)}, remove_columns="audio")

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.dataset = self.dataset.with_format("torch", device=device)

    def dataloader_batch_sampler(self, ds, batch_size):
        batch_sampler = BatchSampler(RandomSampler(
            ds), batch_size=batch_size, drop_last=False)
        return DataLoader(ds, batch_sampler=batch_sampler, collate_fn=self.collate)

    def train_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["train"], self.batch_size)

    def val_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["validation"], self.batch_size)

    def test_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["test"], self.batch_size)
