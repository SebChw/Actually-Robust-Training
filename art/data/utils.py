from torch.utils.data import DataLoader
import lightning as L

import torch
import datasets
import typing


def dummy_classification_sample(shape=(1, 16000), label=0):
    return {"data": torch.randn(shape), "label": label}


def dummy_source_separation_sample(
    shape=(2, 44100), instruments=["bass", "drums", "vocals", "other"]
):
    # We must do it in this way because hydra by default calls functions and doesn't pass them as arguments
    # I don't know if it is possible to pass function as an argument in hydra actually
    # When I tried to pass it as argument it was pased as a string
    return lambda i: {
        **{name: {"array": torch.randn(shape)} for name in instruments},
        "mean": 0,
        "std": 1,
        "name": str(i % 4),
        "n_window": i // 4,
    }


def dummy_generator(sample_gen: typing.Callable, n_samples=64):
    def generator():
        for i in range(n_samples):
            yield sample_gen(i)

    return generator


class SanityCheckDataModule(L.LightningDataModule):
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

        self.dataset = self.dataset.with_format("torch", device="cpu")

    def _get_loader(self):
        return DataLoader(self.dataset, batch_size=4, collate_fn=self.collate)

    def train_dataloader(self):
        return self._get_loader()

    def val_dataloader(self):
        return self._get_loader()

    def test_dataloader(self):
        return self._get_loader()
