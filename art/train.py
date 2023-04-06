from pathlib import Path
from unittest.mock import MagicMock

import hydra
import torch
from lightning import seed_everything
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig, OmegaConf

from art.utils.loggers import get_pylogger

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

    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger = hydra.utils.instantiate(cfg.logger)

    # push configuration
    logger.experiment["configuration"] = stringify_unsupported(
        OmegaConf.to_container(cfg, resolve=True)
    )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Init lightning model
    log.info(f"Instantiating lightning model <{cfg.module._target_}>")
    model = hydra.utils.instantiate(cfg.module)

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

    # Upload logs
    hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])  # type: ignore
    logger.experiment["logs"].upload(str(hydra_dir / "train.log"))

    # Save best checkpoint to the hub
    if cfg.upload_best_model:
        logger.experiment["model_checkpoints/best_model"].upload(
            trainer.checkpoint_callback.best_model_path
        )


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
