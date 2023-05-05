import neptune


class NeptuneApiWrapper:
    def __init__(self, project_name='skdbmk/sourceseparation', mode="read-only"):
        self.project_name = project_name
        self.project = neptune.init_project(project=self.project_name, mode=mode)

    def get_runs_table(self, owner=None, tags=None):
        runs_table_df = self.project.fetch_runs_table(owner=owner, tags=tags).to_pandas()
        return runs_table_df
    
    def get_checkpoint(self, run_id=None, path='./'):
        # None defaults to last run, but including the read only run!
        if "ckpt" not in path:
            path = f"{path}{self.run_id}.ckpt"
        try:
            model_path = self.run["training/model/best_model_path"].fetch().split("/")[-1][:-5]
        except neptune.exceptions.MissingFieldException as e:
            raise Exception(
                f"Couldn't find Best model under specified id {self.run_id}"
            ).with_traceback(e.__traceback__)

        self.run[f"training/model/checkpoints/{model_path}"].download(path)
        return path


if __name__ == "__main__":
    neptune_api = NeptuneApiWrapper()
    neptune_api.get_checkpoint(run_id='SOUR-33')