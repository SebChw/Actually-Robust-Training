import argparse
import neptune
CONFIG_FILE = Path("config.yaml")


class NeptuneApiWrapper:
    def __init__(self, project_name='skdbmk/sourceseparation', mode="read-only"):
        self.project_name = project_name
        self.project = neptune.init_project(project=self.project_name, mode=mode)

    def get_runs_table(self, owner=None, tags=None):
        runs_table_df = self.project.fetch_runs_table(owner=owner, tags=tags).to_pandas()
        return runs_table_df

    def get_checkpoint(self, run_id=None, path='./'):
        # None defaults to last run, but including the read only run!
        run = neptune.init_run(project=self.project_name, with_id=run_id)
        if 'ckpt' not in path:
            path = f'{path}{run_id}.ckpt'
        try:
            model_path = run['model/best_model_path'].fetch().split('/')[-1][:-5]
        except neptune.exceptions.MissingFieldException:
            print("Wrong id!")

        run[f'model/checkpoints/{model_path}'].download(path)
        return run

    def get_config(self, run_id=None):
        run = neptune.init_run(project=self.project_name, with_id=run_id)
        config = run['configuration'].fetch()
        return config


def get_overrides(args):
    overrides = {}
    for arg in args:
        key, value = arg.split("=")
        if key[0] == '+':
            key = key[1:]
        overrides[key] = value
    return overrides


def update_config(cfg, key, value):
    print(key, value)
    if '.' in key:
        key, rest = key.split('.', 1)
        update_config(cfg[key], rest, value)
    else:
        if value == 'null':
            value = None
        cfg[key] = value
        print(cfg)
    return cfg


def get_last_training_data(cfg):
    run_id = cfg.continue_training_id
    neptuneAPIwrapper = NeptuneApiWrapper(project_name=cfg.logger.project)
    neptuneAPIwrapper.get_checkpoint(run_id=run_id, path='./')

    # overrides = get_overrides(sys.argv[1:])

    # cfg = DictConfig(neptuneAPIwrapper.get_config(run_id=run_id))
    cfg.ckpt_path = f"{run_id}.ckpt"
    # for key, value in overrides.items():
    #     cfg = update_config(cfg, key, value)



def push_configuration(logger, cfg: DictConfig):
    # I considered using tempfile but we want to have specific name of the file
    OmegaConf.save(cfg, CONFIG_FILE)
    logger.experiment["config"].upload(str(CONFIG_FILE), wait=True)
    CONFIG_FILE.unlink()
