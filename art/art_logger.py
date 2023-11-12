from contextlib import contextmanager
import sys
from loguru import logger
from pathlib import Path
from datetime import datetime
import uuid
from art.paths import LOG_PATH
logger.remove()
logger.add(sys.stdout, format="{message}", level="DEBUG", filter= lambda record: ('halt_exception' not in record['extra']) or (not record['extra']['halt_exception']))


def get_new_log_file_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid.uuid4()) + ".log"

def add_logger(log_file_path: Path) -> int:
    return art_logger.add(log_file_path, format="{time} {level} {message}", level="DEBUG")

def remove_logger(logger_id: int):
    art_logger.remove(logger_id)

art_logger = logger
