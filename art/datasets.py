from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dataclasses import dataclass, field
import torchaudio
import pytorch_lightning as pl

from art.data.organizers import google_command_organizer, cat_to_numeric
import art.constants as c


@dataclass
class AudioMetadata:
    sr: int
    snr: int = None


def preloaded_audio_load(row):
    return row['audio'], row['target'], row["sr"]


def classification_load(row):
    waveform, sr = torchaudio.load(row[c.PATH_FIELD])
    return waveform, row[c.TARGET_FIELD], sr


class RawAudioUnifier:
    # Mission of this class is to provide unified form of the audio so that it can be put in a batch
    def __init__(self, expected_sr=16000):
        self.expected_sr = expected_sr

    def __call__(self, waveform, target, sr):

        transform = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=self.expected_sr)

        transformed = transform(waveform)

        return transformed, target, sr


class SpectrogramUnifier:
    pass


class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, load, unify, transform=lambda x, y, z : (x, y)):
        super().__init__()
        self.df = df
        self.load = load
        self.unify = unify
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # TODO: Think if this is optimal solution, or maybe audio should be an object with all metadata and target inside?
        # maybe https://docs.python.org/3/library/dataclasses.html?
        audio, target, sr = self.load(row)
        audio, target, sr = self.unify(audio, target, sr)
        audio, target = self.transform(audio, target, sr)

        return audio, target


class GoogleCommandDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.splits = google_command_organizer(self.data_dir)
        cat_to_numeric(self.splits)
        self.batch_size = batch_size
        self.unifier = RawAudioUnifier()

    def setup(self, stage):
        if stage == "fit":
            self.gc_train = AudioDataset(
                self.splits[c.TRAIN_SPLIT], classification_load, self.unifier)
            self.gc_valid = AudioDataset(
                self.splits[c.VALID_SPLIT], classification_load, self.unifier)

        if stage == "test":
            self.gc_test = AudioDataset(
                self.splits[c.TEST_SPLIT], classification_load, self.unifier)

    def train_dataloader(self):
        return DataLoader(self.gc_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.gc_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.gc_test, batch_size=self.batch_size)
