import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from lightning.pytorch.loggers import NeptuneLogger, WandbLogger
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{message}", level="DEBUG")


def get_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid.uuid4())


def get_new_log_file_name(run_id: str) -> str:
    return f"{run_id}.log"


def add_logger(log_file_path: Path) -> int:
    return art_logger.add(
        log_file_path, format="{time} {level} {message}", level="DEBUG"
    )


def remove_logger(logger_id: int):
    art_logger.remove(logger_id)


art_logger = logger


class NeptuneLoggerAdapter(NeptuneLogger):
    """
    This is a wrapper for LightningLogger for simplifying basic functionalities between different loggers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_config(self, configFile, path: str = "hydra/config"):
        """
        Logs a config file to Neptune.

        Args:
            configFile (str): Path to config file.
            path (str, optional): Path to log config file to. Defaults to "hydra/config".
        """
        self.experiment[path].upload(configFile)

    def log_img(self, image, path: str = "image"):
        """
        Logs an image to Neptune.

        Args:
            image (np.ndarray): Image to log.
            path (str, optional): Path to log image to. Defaults to "image".
        """
        self.experiment[path].upload(image)

    def log_figure(self, figure, path: str = "figure"):
        """
        Logs a figure to Neptune.

        Args:
            figure (Any): Figure to log.
            path (str, optional): Path to log figure to. Defaults to "figure".
        """
        self.experiment[path].upload(figure)

    def download_ckpt(
        self,
        id: str,
        name: Optional[str] = None,
        type: str = "last",
        path: str = "./checkpoints",
    ):
        """
        Downloads a checkpoint from Neptune.

        Args:
            id (str): Run ID.
            name (str, optional): Name of the checkpoint. Defaults to None.
            type (str, optional): Type of the checkpoint. Defaults to "last".
            path (str, optional): Path to download checkpoint to. Defaults to "./checkpoints".

        Raises:
            Exception: If the checkpoint does not exist.
            Exception: If the type is not "last" or "best".

        Returns:
            str: Path to downloaded checkpoint.
        """
        from neptune import init_run
        from neptune.exceptions import MissingFieldException

        if name is None:
            name = f"{id}"
        if "ckpt" not in name:
            name = f"{name}.ckpt"
        run = init_run(with_id=id, mode="read-only")
        if type == "last":
            model_path = "last"
        elif type == "best":
            try:
                model_path = os.path.basename(
                    run["training/model/best_model_path"].fetch()
                )[:-5]
            except MissingFieldException as e:
                raise Exception(
                    f"Couldn't find Best model under specified id {id}"
                ).with_traceback(e.__traceback__)
        else:
            raise Exception(f'Unknown type {type}. Use "last" or "best".')

        print(model_path)
        run[f"/training/model/checkpoints/{model_path}"].download(f"{path}/{name}")

        run.stop()
        return f"{dir}/{name}.ckpt"

    def stop(self):
        self.run.stop()

    def add_tags(self, tags: Union[List[str], str]):
        """
        Adds tags to the Neptune experiment.

        Args:
            tags (Union[List[str], str]): Tag or list of tags to add.
        """
        if isinstance(tags, str):
            tags = [tags]
        self.experiment.add_tags(tags)


class WandbLoggerAdapter(WandbLogger):
    """
    This is a wrapper for LightningLogger for simplifying basic functionalities between different loggers.
    Logging plots in Wandb supports Plotly only. If you want to log matplotlib figures, you need to convert them to Plotly first or log them as images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import wandb

        self.wandb = wandb

    def log_config(self, configFile: str):
        """
        Logs a config file to Wandb.

        Works only when run as an admin.

        Args:
            configFile (str): Path to config file.
        """
        # yaml_data = open(configFile, 'r')
        # yaml_data = yaml.load(yaml_data, Loader=yaml.SafeLoader)
        self.wandb.save(configFile)

    def log_img(self, image, path: Union[str, np.ndarray] = "image"):
        """
        Logs an image to Wandb.

        Args:
            image (np.ndarray): Image to log.
            path (str, optional): Path to log image to. Defaults to "image"."""
        self.wandb.log({path: self.wandb.Image(image)})

    def log_figure(self, figure, path="figure"):
        """
        Logs a figure to Wandb.

        Args:
            figure (Any): Figure to log.
            path (str, optional): Path to log figure to. Defaults to "figure".
        """
        self.wandb.log({path: figure})

    def add_tags(self, tags: Union[List[str], str]):
        """
        Adds tags to the Wandb run.

        Args:
            tags (Union[List[str], str]): Tag or list of tags to add.
        """
        if isinstance(tags, str):
            tags = [tags]
        self.wandb.run.tags += tags
