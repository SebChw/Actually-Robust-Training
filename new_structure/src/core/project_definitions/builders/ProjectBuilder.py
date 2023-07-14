class ProjectBuilder:
    def __init__(self, args):
        self.args = args

    def build(self, project_path: str):
        self._build_project_structure(project_path)
        self._build_project_files(project_path)

    def _build_project_structure(self, project_path: str):
        pass

    def _build_project_files(self, project_path: str):
        pass
