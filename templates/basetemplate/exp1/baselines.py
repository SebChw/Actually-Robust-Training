import numpy as np
from datasets import Dataset
from torchmetrics import Accuracy

from art.core.base_components.BaseModel import Baseline


class MlBaseline(Baseline):
    def __init__(self, model):
        super().__init__(accelerator="cpu")
        self.model = model
        self.name = "MlBaseline"

    # prepare_data name is taken by lightning
    def create_data(self, dataloader):
        # TODO it should always return X,y as numpy arrays
        # Taka generalna funkcje prepare data dla MlBaseline powinnismy przygotowac
        X = []
        y = []
        for batch in dataloader:
            X.append(batch["x"])
            y.append(batch["y"])

        return np.concatenate(X), np.concatenate(y)

    # train has conflicts with lightning
    def train_baseline(self, train_data):
        # raise NotImplementedError

        #! Idea jest taka, ze w tym miejscu kazdy dostaje X, y i robi sobie z nim zo zechce
        X, y = self.create_data(train_data)

        # TO jest customowy kod napisany przez usera
        self.model = self.model.fit(X, y)

    def validation_step(self, batch, batch_idx):
        # raise NotImplementedError
        # TODO: even in MLModel we should do this in batch mode.
        x, y = batch["x"], batch["y"]

        predictions = self.model.predict(x)

        # TODO We should probably make all used metrics somehow reusable within different stages.
        self.metric(self, predictions, y)


class HeuristicBaseline(Baseline):
    def __init__(self):
        super().__init__(accelerator="cpu")
        self.name = "HeuristicBaseline"

    def train_baseline(self, train_data):
        """Initialize already existing solution"""
        pass

    def validation_step(self, batch, batch_idx):
        # raise NotImplementedError
        x, y = batch["x"], batch["y"]

        predictions = np.full(x.shape[0], 0)

        # TODO We should probably make all used metrics somehow reusable within different stages.
        self.metric(self, predictions, y)


class AlreadyExistingSolutionBaseline(Baseline):
    def train(self, data):
        """Initialize already existing solution"""

    def validation_step(self, batch, batch_idx):
        """Add it here"""
