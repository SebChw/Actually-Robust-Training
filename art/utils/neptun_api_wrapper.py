import sys
from pathlib import Path

import neptune
from omegaconf import DictConfig, OmegaConf

CONFIG_FILE = Path("config.yaml")


class NeptuneApiWrapper:
    def __init__(self, project_name, run_id):
        self.project_name = project_name
        self.run_id = run_id
        self.run = neptune.init_run(project=self.project_name, with_id=run_id)

    def get_checkpoint(self, path="./"):
        # None defaults to last run, but including the read only run!
        if "ckpt" not in path:
            path = f"{path}{self.run_id}.ckpt"
        try:
            model_path = self.run["model/best_model_path"].fetch().split("/")[-1][:-5]
        except neptune.exceptions.MissingFieldException as e:
            raise Exception(
                f"Couldn't find Best model under specified id {self.run_id}"
            ).with_traceback(e.__traceback__)

        self.run[f"model/checkpoints/{model_path}"].download(path)
        return path


def get_last_run(cfg):
    neptuneAPIwrapper = NeptuneApiWrapper(cfg.logger.project, cfg.continue_training_id)
    neptuneAPIwrapper.run["config"].download()

    cfg = OmegaConf.load(CONFIG_FILE)
    CONFIG_FILE.unlink()

    cfg.ckpt_path = neptuneAPIwrapper.get_checkpoint()

    for config_assignments in sys.argv[1:]:
        key, value = config_assignments.split("=")
        if key[0] == "+":
            key = key[1:]

        if key in ["datamodule", "logger", "module", "trainer"]:
            raise ValueError(
                "You can't overwrite the config of the datamodule, logger, module or trainer as these are composite configs."
            )

        OmegaConf.update(cfg, key, value)

    return neptuneAPIwrapper.run, cfg


def push_configuration(logger, cfg: DictConfig):
    # I considered using tempfile but we want to have specific name of the file
    OmegaConf.save(cfg, CONFIG_FILE)
    logger.experiment["config"].upload(str(CONFIG_FILE), wait=True)
    CONFIG_FILE.unlink()
