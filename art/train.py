from lightning import seed_everything
from art.utils.loggers import get_pylogger
import hydra
from omegaconf import DictConfig
import torch

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "configs",
    "config_name": "train.yaml",
}

log = get_pylogger(__name__)


def train(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # Init lightning model
    log.info(f"Instantiating lightning model <{cfg.module._target_}>")
    model = hydra.utils.instantiate(cfg.module, _recursive_=False)

    if cfg.compile:
        log.info("Compiling the model.")
        model = torch.compile(model, mode="reduce-overhead")

    # Overfit one batch if wanted for sanity check
    if cfg.get("overfit_one_batch"):
        log.info("Overfitting on one batch of the data")
        trainer = hydra.utils.instantiate(cfg.trainer, overfit_batches=1, max_epochs=50)
        trainer.fit(model=model, datamodule=datamodule)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoints)
    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=[checkpoint_callback]
    )

    # Train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    # Test the model
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    # Save state dicts for best and last checkpoints
    if cfg.get("save_state_dict"):
        log.info("Starting saving state dicts!")


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
    # ! Previously it was like that. Now everything is set using config files
    # neptune_logger = get_logger("skdbmk/sourceseparation", ["training", "wwd"])
    # model = networks.M5()
    # pl_module = litmodules.LitAudioClassifier(model, num_classes=36)
    # data_module = datamodules.GoogleCommandDataModule()
    # trainer = L.Trainer(accelerator="gpu", devices=1, logger=neptune_logger)
    # trainer.fit(pl_module, datamodule=data_module)
