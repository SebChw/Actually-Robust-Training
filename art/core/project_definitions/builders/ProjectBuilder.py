class ProjectBuilder:
    def __init__(self, args):
        self.args = args

    def build(self, project_path: str):
        """
        Build project.

        Args:
            project_path (str): Path to project.
        """
        self._build_project_structure(project_path)
        self._build_project_files(project_path)

    def _build_project_structure(self, project_path: str):
        """
        A function that is used first to build the project structure, before any files are added.

        Args:
            project_path (str): Path to project.
        """
        pass

    def _build_project_files(self, project_path: str):
        """
        A function that is used to build the project files.

        Args:
            project_path (str): Path to project.
        """
        pass
