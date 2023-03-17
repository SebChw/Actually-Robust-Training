from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch
import datasets

from torch.utils.data.sampler import BatchSampler, RandomSampler

from art.data.collate import create_waveform_collate, create_sourceseparation_collate


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
        # If you don't write anything here Lightning will try to use prepare_data_per_node
        print("Nothing to download")

    def setup(self, stage):
        # This separation between assignment and donwloading may be usefull for multi-node training
        if self.dataset is None:
            self.dataset = datasets.load_dataset("speech_commands", "v0.02")
            self.dataset = self.dataset.remove_columns(
                ["file", "is_unknown", "speaker_id", "utterance_id"]
            )

            # This is very inefficient it stores stuff on 64 bits by default. HF doesn't support 16 bits.
            # self.dataset = self.dataset.map(
            #     lambda example: {"data": np.expand_dims(example["audio"]["array"], 0)}, remove_columns="audio")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dataset = self.dataset.with_format("torch", device=device)

    def dataloader_batch_sampler(self, ds, batch_size):
        batch_sampler = BatchSampler(
            RandomSampler(ds), batch_size=batch_size, drop_last=False
        )
        return DataLoader(ds, batch_sampler=batch_sampler, collate_fn=self.collate)

    def train_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["train"], self.batch_size)

    def val_dataloader(self):
        return self.dataloader_batch_sampler(
            self.dataset["validation"], self.batch_size
        )

    def test_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["test"], self.batch_size)


class SounDemixingChallengeDataModule(pl.LightningDataModule):
    def __init__(self, zip_path, batch_size=64, spectrogram=False, type_="labelnoise"):
        super().__init__()

        self.zip_path = zip_path
        self.type_ = type_

        if self.type_ == "labelnoise":
            # This is approximatelly 80% of data and also no overlap between train and test set. Given 10s window size!
            self.train_size = 3722

        self.batch_size = batch_size
        self.spectrogram = spectrogram

        self.collate = create_sourceseparation_collate()

        self.dataset = None

    def prepare_data(self):
        print("Nothing to download")

    def setup(self, stage):
        if self.dataset is None:
            self.dataset = datasets.load_dataset(
                "sebchw/sound_demixing_challenge",
                self.type_,
                zip_path=self.zip_path,
                split="train",
            )

            # We don't shuffle not to mix up song between sets
            self.dataset = self.dataset.train_test_split(
                train_size=self.train_size, shuffle=False
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dataset = self.dataset.with_format("torch", device=device)

    def dataloader_batch_sampler(self, ds, batch_size):
        batch_sampler = BatchSampler(
            RandomSampler(ds), batch_size=batch_size, drop_last=False
        )
        return DataLoader(ds, batch_sampler=batch_sampler, collate_fn=self.collate)

    def train_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["train"], self.batch_size)

    def val_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["test"], self.batch_size)
