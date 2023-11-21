from pathlib import Path

CHECKPOINTS_PATH = Path("art_checkpoints")
EXPERIMENT_LOG_DIR = CHECKPOINTS_PATH / "experiment" / "logs"


def get_checkpoint_step_dir_path(full_step_name) -> Path:
    """
    Get the name of the directory for the given step.

    Args:
        step_id (str): The ID of the step.
        step_name (str): The name of the step.

    Returns:
        str: The name of the directory.
    """
    return CHECKPOINTS_PATH / f"{full_step_name}"


def get_checkpoint_logs_folder_path(full_step_name) -> Path:
    return get_checkpoint_step_dir_path(full_step_name) / "logs"


#
EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
