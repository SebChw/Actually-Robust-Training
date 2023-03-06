from art import datamodules, litmodules, networks
import pytorch_lightning as pl
from loggers import get_logger

if __name__ == '__main__':
    model = networks.M5()
    pl_module = litmodules.LitAudioClassifier(model, num_classes=36)
    data_module = datamodules.GoogleCommandDataModule()

    neptune_logger = get_logger(
        "skdbmk/wake-word-detection", ["training", "wwd"])
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=neptune_logger)
    trainer.fit(pl_module, datamodule=data_module)
