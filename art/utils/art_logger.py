import sys
from loguru import logger
from pathlib import Path
from datetime import datetime
import uuid

logger.remove()
logger.add(sys.stdout, format="{message}", level="DEBUG")


def get_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid.uuid4())


def get_new_log_file_name(run_id: str) -> str:
    return f"{run_id}.log"


def add_logger(log_file_path: Path) -> int:
    return art_logger.add(log_file_path, format="{time} {level} {message}", level="DEBUG")


def remove_logger(logger_id: int):
    art_logger.remove(logger_id)


art_logger = logger
