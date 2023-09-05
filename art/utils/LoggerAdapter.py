from lightning.pytorch.loggers import NeptuneLogger, WandbLogger
import wandb
# from matplotlib.figure import Figure
from typing import Union, Type
import numpy as np
import os
import neptune



# """
#     This is a wrapper for LightningLogger for simplifying basic functionalities between different loggers.
#     Logging plots in Wandb supports Plotly only. If you want to log matplotlib figures, you need to convert them to Plotly first or log them as images.
# """

class NeptuneLoggerAdapter(NeptuneLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_config(self, configFile, path: str = "hydra/config"):
        self.experiment[path].upload(configFile)

    # def log_model(self, model: LightningModule, path: str = "model"):
    #     self.experiment[path].track_files(model)

    def log_image(self, image, path: str = "image"):
        self.experiment[path].upload(image)

    def log_figure(self, figure, path: str = "figure"):
        self.experiment[path].upload(figure)

    def download_ckpt(self, id: str, name: str = "", type: str = "last", path: str = "./checkpoints"):
        if name == "" or name is None:
            name = f"{id}"
        if "ckpt" not in name:
            name = f"{name}.ckpt"
        run = neptune.init_run(with_id = id, mode="read-only")
        if type == "last":
            model_path = "last"
        elif type == "best":
            try:
                model_path = (
                    run["training/model/best_model_path"].fetch().split("\\")[-1][:-5] # change "\\" to "/" if you are using Linux
                )
            except neptune.exceptions.MissingFieldException as e:
                raise Exception(
                    f"Couldn't find Best model under specified id {id}"
                ).with_traceback(e.__traceback__)
        else:
            raise Exception(f"Unknown type {type}. Use \"last\" or \"best\".")

        print(model_path)
        run[f"/training/model/checkpoints/{model_path}"].download(f"{path}/{name}")

        run.stop()
        return f"{dir}/{name}.ckpt"



class WandbLoggerAdapter(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_config(self, configFile: str):
        """Works only when run as an admin."""
        # yaml_data = open(configFile, 'r')
        # yaml_data = yaml.load(yaml_data, Loader=yaml.SafeLoader)
        wandb.save(configFile)

    def log_image(self, image, path: Union[str, np.ndarray] = "image"):
        wandb.log({path: wandb.Image(image)})

    def log_figure(self, figure, path = "figure"):
        wandb.log({path: figure})
