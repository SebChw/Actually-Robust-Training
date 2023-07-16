from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from src.core.base_components.BaseModel import Baseline


class MlBaseline(Baseline):
    def prepare_data(self, data):
        # TODO it should always return pandas dataframe
        # Taka generalna funkcje prepare data dla MlBaseline powinnismy przygotowac
        if isinstance(data, Dataset):
            return data.to_pandas()

    def train(self, train_data):
        # raise NotImplementedError

        #! Idea jest taka, ze w tym miejscu kazdy dostaje pandasa i robi sobie z nim zo zechce
        train_data = self.prepare_data(train_data)

        # TO jest customowy kod napisany przez usera
        target_col = train_data.columns == "y"
        X, y = train_data.loc[:, ~target_col], train_data.loc[:, target_col]
        self.model = LogisticRegression().fit(X, y)

    def validation_step(self, batch, batch_idx):
        # raise NotImplementedError
        # TODO: even in MLModel we should do this in batch mode.
        x, y = self.prepare_batch(batch)

        predictions = self.model(X)

        # TODO We should probably make all used metrics somehow reusable within different stages.
        # self.accurac(predictions, y)


class HeuristicBaseline(Baseline):
    def validation_step(self, batch, batch_idx):
        # raise NotImplementedError

        x, y = self.prepare_batch(batch)

        predictions = np.full(x.shape[0], 5)

        # TODO We should probably make all used metrics somehow reusable within different stages.
        # self.accurac(predictions, y)


class AlreadyExistingSolutionBaseline(Baseline):
    def train(self, data):
        """Initialize already existing solution"""

    def validation_step(self, batch, batch_idx):
        """Add it here"""
