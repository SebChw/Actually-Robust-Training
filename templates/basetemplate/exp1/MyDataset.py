import lightning.pytorch as pl
from datasets import Dataset
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader

from art.core.base_components.BaseDataset import BaseDataset

# I moved it here so that imports works. We must agree on how we handle this.


class MyDataset(BaseDataset):
    pass


# I wonder if we need anything more than LightningDataModule
class DummyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        X, y = make_classification(n_samples=10000)
        self.dataset = Dataset.from_dict({"x": X, "y": y})
        self.dataset = self.dataset.with_format("torch")

    def prepare_data_per_node(
        self,
    ):
        pass

    def setup(self, stage: str):
        pass

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=16)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=16)

    def turn_on_regularization(self):
        """But in case of dataset it is probably better to put this on the user."""
        return self

    def turn_off_regularization(self):
        pass
