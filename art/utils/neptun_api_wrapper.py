import argparse
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
        run = neptune.init_run(project=self.project_name, with_id=run_id)
        if 'ckpt' not in path:
            path = f'{path}{run_id}.ckpt'
        try: 
            model_path = run['model/best_model_path'].fetch().split('/')[-1][:-5]
        except neptune.exceptions.MissingFieldException:
            print("Wrong id!")

        run[f'model/checkpoints/{model_path}'].download(path)
        return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neptune API Wrapper')
    parser.add_argument('--project_name', type=str, default='skdbmk/sourceseparation', help='Neptune project name')
    parser.add_argument('--run_id', type=str, default='SOUR-33', help='Run ID for checkpoint')
    args = parser.parse_args()

    neptune_api = NeptuneApiWrapper(project_name=args.project_name)
    neptune_api.get_checkpoint(run_id=args.run_id)
