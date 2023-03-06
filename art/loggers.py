from pytorch_lightning.loggers import NeptuneLogger


def get_logger(project_name, tags):
    return NeptuneLogger(
        project=project_name,
        tags=tags,  # optional
    )
