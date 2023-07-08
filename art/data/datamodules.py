import datasets
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

from art.data.collate import SourceSeparationCollate


class SourceSeparationDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_kwargs,
        batch_size=64,
        train_size=None,
        max_length=-1,
        num_workers=4,
        dataset_path=None,
        augment=False,
    ):
        super().__init__()

        self.train_size = train_size
        self.dataset_path = dataset_path
        self.dataset_kwargs = dataset_kwargs

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.collate = SourceSeparationCollate(max_length, augment)
        self.dataset = None

    def prepare_data(self):
        print("Nothing to download")

    def setup(self, stage):
        if self.dataset is None:
            if self.dataset_path is not None:
                print(f"Loading dataset {self.dataset_path} from")
                self.dataset = datasets.DatasetDict.load_from_disk(self.dataset_path)
            else:
                self.dataset = datasets.load_dataset(**self.dataset_kwargs)

                if self.dataset_kwargs["path"] == "sebchw/sound_demixing_challenge":
                    self.dataset["test"] = datasets.load_dataset(
                        "sebchw/musdb18", split="test"
                    )

            self.dataset = self.dataset.with_format("torch", device="cpu")

    def dataloader_batch_sampler(self, ds, batch_size):
        batch_sampler = BatchSampler(
            RandomSampler(ds), batch_size=batch_size, drop_last=False
        )
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=self.collate,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["train"], self.batch_size)

    def val_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["test"], self.batch_size)

    def test_dataloader(self):
        return self.dataloader_batch_sampler(self.dataset["test"], self.batch_size)
