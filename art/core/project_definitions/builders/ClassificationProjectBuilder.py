from .ProjectBuilder import ProjectBuilder


class ClassificationProjectBuilder(ProjectBuilder):
    """A builder for classification projects."""

    def __init__(self, args):
        super().__init__(args)
