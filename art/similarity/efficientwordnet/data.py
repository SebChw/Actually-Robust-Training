import os
import random
from os import listdir, mkdir
from os.path import isdir

import librosa
import lightning as L
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from art.enums import TrainingStage

AUDIO_LENGTH = 16000

"""This is code from https://github.com/Ant-Brain/EfficientWord-Net/blob/main/training.ipynb Written for This framework"""


def _random_crop(x, length=AUDIO_LENGTH):
    assert x.shape[1] > length
    frontBits = random.randint(0, x.shape[1] - length)
    return x[:, frontBits : frontBits + length]


#!All right they randomly does it here. They Add random number of 0 to the fron at back
#! But again it may be different each time if it's
def _add_padding(x, length=AUDIO_LENGTH):
    assert x.shape[1] < length
    bitCountToBeAdded = length - x.shape[1]
    frontBits = random.randint(0, bitCountToBeAdded)

    new_x = np.append(np.zeros(frontBits), x)
    new_x = np.append(new_x, np.zeros(bitCountToBeAdded - frontBits))
    return F.pad(x, (frontBits, bitCountToBeAdded - frontBits), "constant", 0)


#! There is much more efficient solution for this proposed by numpy.trim_zeros
#! But now the question is whether doing so is always beneficial or not. We should give model a chance to hear
#!Silence, noise withouth speech etc.
#! Moreover this is highly inneficieny as it is done for every single audio file every epoch.
def _remove_existing_padding(x: np.ndarray) -> np.ndarray:
    lastZeroBitBeforeAudio = 0
    firstZeroBitAfterAudio = len(x[0])
    for i in range(len(x[0])):
        # They use numpy arrays so 0th index is necessary as we use torchaudio
        if x[0][i] == 0:
            lastZeroBitBeforeAudio = i
        else:
            break
    for i in range(len(x[0]) - 1, 1, -1):
        if x[0][i] == 0:
            firstZeroBitAfterAudio = i
        else:
            break
    return x[:, lastZeroBitBeforeAudio:firstZeroBitAfterAudio]


def _fix_padding_issues(x: np.ndarray, length=AUDIO_LENGTH) -> np.ndarray:
    x = _remove_existing_padding(x)
    # x = randomAugumentation(x)
    # print("Preprocessing Shape",x.shape[0])
    if x.shape[1] > length:
        return _random_crop(x, length=length)
    elif x.shape[1] < length:
        return _add_padding(x, length=length)
    else:
        return x


def _omit_hidden_files(inpArray):
    return [x for x in inpArray if "." != x[0]]


class EfficientWordNetOriginalDataset(Dataset):
    def __init__(
        self,
        chunkedNoisePath,
        datasetPath,
        stage: TrainingStage,
        max_noise_factor=0.2,
        min_noise_factor=0.05,
    ):
        self.chunkedNoisePath = chunkedNoisePath
        self.datasetPath = datasetPath
        self.max_noise_factor = max_noise_factor
        self.min_noise_factor = min_noise_factor
        self.sr = 16000
        self.types_of_noise = _omit_hidden_files(os.listdir(self.chunkedNoisePath))
        self.words_in_dataset = _omit_hidden_files(os.listdir(self.datasetPath))

        count_of_words = len(self.words_in_dataset)
        if stage == TrainingStage.TRAIN:
            self.words_in_dataset = self.words_in_dataset[: int(0.9 * count_of_words)]
        else:
            self.words_in_dataset = self.words_in_dataset[int(0.9 * count_of_words) :]

        # Magic value but it makes sense
        self.n = 2 * len(self.words_in_dataset)

        # Our mel spectrogram produces 64x101 compared to their 64x98
        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=400, hop_length=160, n_mels=64, normalized=True
        )

        self.resamplers = {}

    def _resample_if_needed(self, x, sample_sr):
        if sample_sr != self.sr:
            if sample_sr not in self.resamplers:
                self.resamplers[sample_sr] = torchaudio.transforms.Resample(
                    sample_sr, self.sr
                )
            x = self.resamplers[sample_sr](x)
        return x

    #! I don't know why they do this max scaling. Probably standarization would be better
    def _add_noise(self, x, noise, noise_factor=0.4):
        assert x.shape[0] == noise.shape[0]
        out = (1 - noise_factor) * x / x.max() + noise_factor * (noise / noise.max())
        return out / out.max()

    def _give_joined_audio(self, word1, word2):
        if word1 == word2:
            #! Again this is a random process and we don't set seed
            sample1, sample2 = random.sample(
                _omit_hidden_files(os.listdir(self.datasetPath + "/" + word1)), 2
            )
        else:
            sample1 = random.choice(
                _omit_hidden_files(os.listdir(self.datasetPath + "/" + word1))
            )
            sample2 = random.choice(
                _omit_hidden_files(os.listdir(self.datasetPath + "/" + word2))
            )

        #! torchaudio doesn't support automatic resampling
        #! Samples are not neceserily 1 second long
        voiceVector1, sr1 = torchaudio.load(
            self.datasetPath + "/" + word1 + "/" + sample1
        )
        voiceVector2, sr2 = torchaudio.load(
            self.datasetPath + "/" + word2 + "/" + sample2
        )

        voiceVector1 = self._resample_if_needed(voiceVector1, sr1)
        voiceVector2 = self._resample_if_needed(voiceVector2, sr2)

        voiceVector1 = _fix_padding_issues(voiceVector1)
        voiceVector2 = _fix_padding_issues(voiceVector2)

        randomNoiseType1, randomNoiseType2 = random.sample(self.types_of_noise, 2)

        randomNoise1 = random.choice(
            _omit_hidden_files(
                os.listdir(self.chunkedNoisePath + "/" + randomNoiseType1 + "/")
            )
        )
        randomNoise2 = random.choice(
            _omit_hidden_files(
                os.listdir(self.chunkedNoisePath + "/" + randomNoiseType2 + "/")
            )
        )

        noiseVector1, sr1 = torchaudio.load(
            self.chunkedNoisePath + "/" + randomNoiseType1 + "/" + randomNoise1,
        )
        noiseVector2, sr2 = torchaudio.load(
            self.chunkedNoisePath + "/" + randomNoiseType2 + "/" + randomNoise2,
        )

        noiseVector1 = self._resample_if_needed(noiseVector1, sr1)
        noiseVector2 = self._resample_if_needed(noiseVector2, sr2)

        randomNoiseFactor1 = random.uniform(
            self.min_noise_factor, self.max_noise_factor
        )
        randomNoiseFactor2 = random.uniform(
            self.min_noise_factor, self.max_noise_factor
        )

        voice_with_noise1 = self._add_noise(
            voiceVector1, noiseVector1, randomNoiseFactor1
        )
        voice_with_noise2 = self._add_noise(
            voiceVector2, noiseVector2, randomNoiseFactor2
        )

        return self.spec_transform(voice_with_noise1), self.spec_transform(
            voice_with_noise2
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # every word is paired oce with same word and once with word next to it. Weird but they did it like that
        # self.n//2 Is usednot to jump over all words
        audio1, audio2 = self._give_joined_audio(
            self.words_in_dataset[idx // 2],
            self.words_in_dataset[(idx // 2 + idx % 2) % (self.n // 2)],
        )

        y = torch.tensor([1.0]) if idx % 2 == 0 else torch.tensor([0.0])
        return {"X1": audio1, "X2": audio2, "y": y}


class EfficientWordNetOriginalDataModule(L.LightningDataModule):
    def __init__(self, noise_chunked_path, noise_path, words_path, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        self.chunked_noise_path = noise_chunked_path
        self.noise_path = noise_path
        self.words_path = words_path

    def split_noise_file_to_chunks(
        self, filename: str, target_folder: str, count=100, sr=16000
    ):
        # print(filename)
        noiseAudio, _ = librosa.load(filename, sr=sr)
        # They cut randomly 100 chunks out of the noise Audio file
        # If one will try to reproduce it every time it will be different
        for i in range(count):
            noiseAudioCrop = _random_crop(np.expand_dims(noiseAudio, 0))[0]
            outFilePath = (
                target_folder
                + "/"
                + (f"{'.'.join(filename.split('.')[:-1])}_{i}.wav").split("/")[-1]
            )
            # print(filename,outFilePath)
            sf.write(outFilePath, noiseAudioCrop, sr, "PCM_24")
            count = count + 1

    def split_noise_files(self):
        noise_categories = _omit_hidden_files(listdir(self.noise_path))
        for i, noise_category in enumerate(noise_categories):
            source_path = self.noise_path + "/" + noise_category
            target_path = self.chunked_noise_path + "/" + noise_category

            if isdir(source_path):
                mkdir(target_path)
                audioFiles = listdir(source_path)

                for j, audioFile in enumerate(audioFiles):
                    srcFilePath = f"{source_path}/{audioFile}"
                    self.split_noise_file_to_chunks(srcFilePath, target_path)

    def prepare_data(self):
        if isdir(self.chunked_noise_path):
            return

        mkdir(self.chunked_noise_path)
        self.split_noise_files()

    def setup(self, stage):
        # This separation between assignment and donwloading may be usefull for multi-node training
        if not hasattr(self, "dataset_train"):
            self.dataset_train = EfficientWordNetOriginalDataset(
                self.chunked_noise_path, self.words_path, TrainingStage.TRAIN
            )
            self.dataset_val = EfficientWordNetOriginalDataset(
                self.chunked_noise_path, self.words_path, TrainingStage.VALIDATION
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            shuffle=True,
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
