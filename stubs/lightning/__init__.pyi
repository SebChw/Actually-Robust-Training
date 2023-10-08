from .pytorch.core.datamodule import LightningDataModule as LightningDataModule
from .pytorch.core.module import LightningModule as LightningModule
from .utilities.seed import seed_everything as seed_everything

from typing import Dict, Any


class Stage:
    def __init__(self):
        self.value = ""

class State:
    def __init__(self):
        self.stage = Stage()

class Trainer:

    def __init__(self, max_epochs: int =1000, logger: Any ="logger", accelerator: str ="", overfit_batches: float=0.0):
        self.logged_metrics: Dict = {}
        self.state = State()
        self.logger = logger
    def fit(self, model: LightningModule, datamodule: LightningDataModule):
        ...
    
    def validate(self, model: LightningModule, datamodule: LightningDataModule):
        ...

    def test(self, model: LightningModule, datamodule: LightningDataModule):
        ...

    def tune(self, model: LightningModule, datamodule: LightningDataModule):
        ...
    