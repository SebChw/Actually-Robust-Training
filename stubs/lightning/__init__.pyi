from typing import Any, Dict, Optional

from .pytorch.core.callbacks import ModelCheckpoint
from .pytorch.core.datamodule import LightningDataModule as LightningDataModule
from .pytorch.core.module import LightningModule as LightningModule
from .utilities.seed import seed_everything as seed_everything

class Stage:
    def __init__(self):
        self.value = ""

class State:
    def __init__(self):
        self.stage = Stage()

class Trainer:
    model: LightningModule
    checkpoint_callback: ModelCheckpoint
    accelerator: str
    def __init__(
        self,
        max_epochs: int = 1000,
        logger: Any = "logger",
        accelerator: str = "",
        overfit_batches: float = 0.0,
    ):
        self.logged_metrics: Dict = {}
        self.state = State()
        self.logger = logger
    def fit(
        self, model: Optional[LightningModule], datamodule: LightningDataModule
    ): ...
    def validate(
        self, model: Optional[LightningModule], datamodule: LightningDataModule
    ): ...
    def test(
        self, model: Optional[LightningModule], datamodule: LightningDataModule
    ): ...
    def tune(
        self, model: Optional[LightningModule], datamodule: LightningDataModule
    ): ...
