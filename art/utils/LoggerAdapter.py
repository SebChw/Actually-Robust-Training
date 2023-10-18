import os
from typing import Optional, Union

from neptune import init_run
from neptune.exceptions import MissingFieldException
import numpy as np
from wandb import log, save, Image
from lightning.pytorch.loggers import NeptuneLogger, WandbLogger


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


class WandbLoggerAdapter(WandbLogger):
    """
    This is a wrapper for LightningLogger for simplifying basic functionalities between different loggers.
    Logging plots in Wandb supports Plotly only. If you want to log matplotlib figures, you need to convert them to Plotly first or log them as images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_config(self, configFile: str):
        """
        Logs a config file to Wandb.

        Works only when run as an admin.

        Args:
            configFile (str): Path to config file.
        """
        # yaml_data = open(configFile, 'r')
        # yaml_data = yaml.load(yaml_data, Loader=yaml.SafeLoader)
        save(configFile)

    def log_img(self, image, path: Union[str, np.ndarray] = "image"):
        """
        Logs an image to Wandb.

        Args:
            image (np.ndarray): Image to log.
            path (str, optional): Path to log image to. Defaults to "image"."""
        log({path: Image(image)})

    def log_figure(self, figure, path="figure"):
        """
        Logs a figure to Wandb.

        Args:
            figure (Any): Figure to log.
            path (str, optional): Path to log figure to. Defaults to "figure".
        """
        log({path: figure})
