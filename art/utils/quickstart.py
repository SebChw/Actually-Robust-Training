from typing import Any

import datasets
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from art.core import ArtModule
from art.utils.enums import BATCH, INPUT, LOSS, PREDICTION, TARGET


def get_model():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, 1, "same"),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(8, 32, 3, 1, "same"),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1568, 10),
    )


class LightningDataModuleExample(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.mnist_data = datasets.load_dataset("fashion_mnist").with_format("torch")
        if isinstance(self.mnist_data, datasets.DatasetDict):
            self.train_set = self.mnist_data["train"].select(range(200))
            self.test_set = self.mnist_data["test"].select(range(200))

        self.n_classes = 10

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: str):
        pass

    def val_dataloader(self):
        return DataLoader(self.train_set, batch_size=16)

    def train_dataloader(self):
        return DataLoader(self.test_set, batch_size=16)

    def log_params(self):
        return {
            "n_classes": self.n_classes,
            "n_train": len(self.train_set),
            "n_test": len(self.test_set),
        }


class ArtModuleExample(ArtModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()
        self.lr = 0.001

    def parse_data(self, data):
        x = data[BATCH]["image"]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float() / 255
        return {INPUT: x, TARGET: data[BATCH]["label"]}

    def predict(self, data):
        return {PREDICTION: self.model(data[INPUT]), **data}

    def compute_loss(self, data):
        loss = data["CrossEntropyLoss"]
        return {LOSS: loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_params(self):
        return {
            "lr": self.lr,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
