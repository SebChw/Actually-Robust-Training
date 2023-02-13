from torch.utils.data import Dataset
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, load, unify, transform=lambda x: x):
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
        audio, target, metadata = self.load(row)
        audio, target, metadata = self.unify(audio, target, metadata)
        audio, target = self.transform(audio, target, metadata)

        return audio, target
