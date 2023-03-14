from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch
import datasets
import typing


def dummy_classification_sample(shape=(1, 16000), label=0):
    return {"data": torch.randn(shape), "label": label}


def dummy_source_separation_sample(
    shape=(2, 44100), instruments=["bass", "drums", "vocals", "other"]
):
    return {name: {"array": torch.randn(shape)} for name in instruments}


def dummy_generator(sample_gen: typing.Callable, n_samples=64, **kwargs):
    def generator():
        for _ in range(n_samples):
            yield sample_gen(**kwargs)

    return generator


class SanityCheckDataModule(pl.LightningDataModule):
    def __init__(self, dataset_generator, collate):
        super().__init__()
        self.dataset_generator = dataset_generator
        self.collate = collate

    def prepare_data(self):
        # If you don't write anything here Lightning will try to use prepare_data_per_node
        print("Nothing to download")

    def setup(self, stage):
        # be aware of caching.
        self.dataset = datasets.Dataset.from_generator(self.dataset_generator)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.dataset.with_format("torch", device=device)

    def _get_loader(self):
        return DataLoader(self.dataset, batch_size=32, collate_fn=self.collate)

    def train_dataloader(self):
        return self._get_loader()

    def val_dataloader(self):
        return self._get_loader()

    def test_dataloader(self):
        return self._get_loader()
