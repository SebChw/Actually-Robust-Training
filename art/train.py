from art import litmodules, networks
import pytorch_lightning as pl
from art.data import datamodules  # , utils, collate
from art.utils.loggers import get_logger

# from torchaudio.models import HDemucs

if __name__ == "__main__":
    print("HEllo world")
    neptune_logger = get_logger("skdbmk/sourceseparation", ["training", "wwd"])

    model = networks.M5()
    pl_module = litmodules.LitAudioClassifier(model, num_classes=36)
    data_module = datamodules.GoogleCommandDataModule()
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=neptune_logger)
    trainer.fit(pl_module, datamodule=data_module)

    # SOURCES = ["bass", "vocals", "drums", "other"]
    # model = HDemucs(SOURCES)
    # pl_module = litmodules.LitAudioSourceSeparator(model, SOURCES)

    # !Sanity Checking
    # data_module = utils.SanityCheckDataModule(
    #     utils.dummy_generator(utils.dummy_source_separation_sample), collate.create_sourceseparation_collate())
    # trainer = pl.Trainer(accelerator="gpu", devices=1)
    # trainer.fit(pl_module, datamodule=data_module)

    # real training

    # neptune_logger = get_logger(
    #     "skdbmk/sourceseparation", ["training", "wwd"])
    # data_module = datamodules.SounDemixingChallengeDataModule(
    #     "sdxdb23_labelnoise_v1.0_rc1.zip", batch_size=2)
    # trainer = pl.Trainer(accelerator="gpu", devices=1, logger=neptune_logger)
    # trainer.fit(pl_module, datamodule=data_module)
