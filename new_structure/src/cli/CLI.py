from enum import Enum


class ProjectType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    CLUSTERING = 3
    TIME_SERIES = 4


class CLI:
    def __init__(self):
        return

    def run_training(self, project_name: str, args):
        pass

    def run_evaluation(self, project_name: str, args):
        pass

    def create_project(self, project_name: str, project_type: ProjectType ,args):
        pass

    def run_checks(self, project_name: str, args):
        pass

    def do_step(self, project_name: str, step_name:str, args):
        pass

