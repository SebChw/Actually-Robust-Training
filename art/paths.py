from pathlib import Path

CHECKPOINTS_PATH = Path("checkpoints")
LOG_PATH = Path("logs")
EXPERIMENT_LOG_PATH = CHECKPOINTS_PATH/"experiment"/"logs"/"logs.log"


def get_checkpoint_step_dir_path(step_id: str, step_name: str) -> Path:
    """
    Get the name of the directory for the given step.

    Args:
        step_id (str): The ID of the step.
        step_name (str): The name of the step.

    Returns:
        str: The name of the directory.
    """
    return CHECKPOINTS_PATH/f"{step_id}_{step_name}"


def get_checkpoint_logs_folder_path(step_id: str, step_name: str) -> Path:
    return get_checkpoint_step_dir_path(step_id, step_name)/"logs"


#mkdirs
EXPERIMENT_LOG_PATH.mkdir(parents=True, exist_ok=True)
