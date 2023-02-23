from pytorch_lightning.loggers import NeptuneLogger
# ! You need to configure environmental variable
from neptune.new import ANONYMOUS_API_TOKEN


def get_logger(project_name, tags):
    return NeptuneLogger(
        api_key=ANONYMOUS_API_TOKEN,
        project=project_name,
        tags=tags,  # optional
    )
