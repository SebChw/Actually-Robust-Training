from art.core import ArtModule
from art.utils.enums import BATCH, PREDICTION

import torch
from torch import nn

from typing import List
from copy import deepcopy


class ArtEnsemble(ArtModule):
    """
    Base class for ensembles.
    """

    def __init__(self, models: List[ArtModule]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def predict(self, data):
        predictions = torch.stack([self.predict_on_model_from_dataloader(model, deepcopy(data)) for model in self.models])
        return torch.mean(predictions, dim=0)

    def predict_on_model_from_dataloader(self, model, dataloader):
        predictions = []
        for batch in dataloader:
            model.to(self.device)
            batch_processed = model.parse_data({BATCH: batch})
            predictions.append(model.predict(batch_processed)[PREDICTION])
        return torch.cat(predictions)

    def log_params(self):
        return {
            "num_models": len(self.models),
            "models": [model.log_params() for model in self.models],
        }
