import datasets
import lightning as L
import torch
from torch.utils.data import DataLoader

from art.data.collate import create_waveform_collate


class GoogleCommandDataModule(L.LightningDataModule):
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

    def train_dataloader(self):
        # Why  Iremoved samplers
        # https://huggingface.co/docs/datasets/use_with_pytorch
        return DataLoader(
            self.dataset["train"].shuffle(),
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"], batch_size=self.batch_size, collate_fn=self.collate
        )
