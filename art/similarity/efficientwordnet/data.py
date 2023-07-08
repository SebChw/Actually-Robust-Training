import random
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchaudio
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset

AUDIO_LENGTH = 16000


def _fix_padding_issues(x, length=AUDIO_LENGTH):
    n_samples = len(x)
    if n_samples > length:  # random crop
        frontBits = random.randint(0, n_samples - length)
        return x[frontBits : frontBits + length]

    needed_pad = length - n_samples
    pad_left = random.randint(0, needed_pad)
    pad_right = needed_pad - pad_left

    return torch.nn.functional.pad(x, (pad_left, pad_right), "constant")


class EfficientWordNetDataset(Dataset):
    def __init__(
        self,
        dataset,
        noise_dataset,
        max_noise=0.2,
        min_noise=0.05,
    ):
        self.dataset = dataset
        self.noise_dataset = noise_dataset
        self.max_noise = max_noise
        self.min_noise = min_noise

        self.label_to_indices = defaultdict(lambda: list())
        for i, label in enumerate(dataset["label"]):
            self.label_to_indices[label.item()].append(i)

        self.label_to_indices[dataset["label"].max().item()] = self.label_to_indices[
            dataset["label"].min().item()
        ]

        self.sr = 16000
        # Our mel spectrogram produces 64x101 compared to their 64x98
        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=400, hop_length=160, n_mels=64, normalized=True
        )

    # ! I don't know why they do this max scaling. Probably standarization would be better
    def _add_noise(self, x, noise, noise_factor=0.4):
        out = (1 - noise_factor) * x / x.max() + noise_factor * (noise / noise.max())
        return out / out.max()

    def __len__(self):
        return len(self.dataset) * 2

    def __getitem__(self, idx):
        # every word is paired oce with same word and once with word next to it. Weird but they did it like that
        n_sample = idx // 2
        same_class = idx % 2 == 0

        audio1 = self.dataset[n_sample]["samples"]
        if same_class:
            indices = self.label_to_indices[self.dataset[n_sample]["label"].item()]
            selected = random.choice(indices)
            while selected == n_sample:
                selected = random.choice(indices)
        else:
            indices = self.label_to_indices[self.dataset[n_sample]["label"].item()]
            selected = random.choice(indices)

        audio2 = self.dataset[selected]["samples"]

        audio1 = _fix_padding_issues(audio1)
        audio2 = _fix_padding_issues(audio2)

        noise1id, noise2id = np.random.choice(len(self.noise_dataset), 2)
        noise1 = self.noise_dataset[noise1id.item()]["audio"]["array"]
        noise2 = self.noise_dataset[noise2id.item()]["audio"]["array"]

        noise_factor1, noise_factor2 = np.random.uniform(
            self.min_noise, self.max_noise, 2
        )

        audio1 = self._add_noise(audio1, noise1, noise_factor1)
        audio2 = self._add_noise(audio2, noise2, noise_factor2)

        audio1 = self.spec_transform(audio1)
        audio2 = self.spec_transform(audio2)

        y = torch.tensor([1.0]) if same_class else torch.tensor([0.0])
        return {
            "X1": torch.unsqueeze(audio1, 0),
            "X2": torch.unsqueeze(audio2, 0),
            "y": y,
        }


def _trim_zeros(example):
    audio = example["audio"]["array"]
    return {"samples": np.trim_zeros(audio).astype(np.float32)}


def _get_chunked_audio(example):
    audio = example["audio"][0]
    waveform = audio["array"]
    sr = audio["sampling_rate"]
    path = Path(audio["path"]).parts

    chunks = []
    for i in range(0, len(waveform) - sr, sr):
        chunks.append(
            {
                "path": f"{path[-2]}_{path[-1][:-4]}_{i//sr}",
                "array": waveform[i : i + sr],
                "sampling_rate": sr,
            }
        )

    return {"audio": chunks}


class EfficientWordNetDataModule(L.LightningDataModule):
    def __init__(self, noise_path, word_path, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        self.noise_path = noise_path
        self.word_path = word_path

    def prepare_data(self):
        pass

    def setup(self, stage):
        # This separation between assignment and donwloading may be usefull for multi-node training
        if not hasattr(self, "dataset_train"):
            dataset = load_dataset("audiofolder", data_dir=self.word_path)["train"]
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            dataset = dataset.map(_trim_zeros, remove_columns=["audio"])
            dataset = dataset.with_format("torch")

            train_data = dataset.select(torch.argwhere(dataset["label"] < 2425))
            test_data = dataset.select(torch.argwhere(dataset["label"] >= 2425))

            noise = load_dataset("audiofolder", data_dir=self.noise_path)["train"]
            noise = noise.cast_column("audio", Audio(sampling_rate=16000))
            noise = noise.map(
                _get_chunked_audio,
                remove_columns=["label", "audio"],
                batched=True,
                batch_size=1,
            )
            noise = noise.with_format("torch")

            self.dataset_train = EfficientWordNetDataset(train_data, noise)
            self.dataset_val = EfficientWordNetDataset(test_data, noise)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_val,
            # shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
        )


class DatasetToCrash(L.LightningModule):
    """Implement dataset to be destroyed"""


    