from loguru import logger
from contextlib import contextmanager
import os
import sys
from pathlib import Path

from art.paths import LOG_PATH

SUPRESS_STANDARD_OUPUT = True

# Configure the logger to log to stdout and a default log file
logger.remove()  # Remove all other handlers
logger.add(sys.stdout, format="{message}", level="DEBUG")
LOG_PATH.mkdir(exist_ok=True)
default_log_file = LOG_PATH / "default.log"
default_handler_id = logger.add(default_log_file, format="{time} {level} {message}", level="DEBUG")


class ArtLogInterceptor:
    """
    A class that intercepts writes to stdout and logs them using a logger.
    """

    def __init__(self, log_file, to_intercept, supress=False):
        self.terminal = to_intercept
        self.supress = supress
        self.log = open(log_file, "a")

    def write(self, message):
        if not self.supress:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        if not self.supress:
            self.terminal.flush()
        self.log.flush()


# Redirect stdout to StdoutInterceptor

interceptor_std = ArtLogInterceptor(LOG_PATH / "stdout.log", sys.stdout, supress=SUPRESS_STANDARD_OUPUT)
sys.stdout = interceptor_std
interceptor_error = ArtLogInterceptor(LOG_PATH / "stderr.log", sys.stderr, supress=True)
sys.stderr = interceptor_error
logger_stack_name = [LOG_PATH / "default.log"]
logger_stack_id = [default_handler_id]


@contextmanager
def log_to_file(file_path: Path):
    """
    A context manager that configures the Loguru logger to write to a specified file.
    """
    if len(logger_stack_name) > 0:
        previous_logger_id = logger_stack_id[-1]
        logger.remove(previous_logger_id)
    file_handler_id = logger.add(str(file_path), format="{time} {level} {message}", level="DEBUG")
    logger_stack_name.append(str(file_path))
    logger_stack_id.append(file_handler_id)
    try:
        yield
    except Exception as e:
        if len(logger_stack_name) >= 0:
            interceptor_error.supress = True
        else:
            interceptor_error.supress = False
        logger.bind(id=logger_stack_id[-1]).exception(e)
        raise e
    finally:
        # Remove the file handler added for this specific logging context
        logger.remove(logger_stack_id[-1])
        logger_stack_name.pop()
        logger_stack_id.pop()
        if len(logger_stack_name) > 0:
            previous_logger_name = logger_stack_name[-1]
            previous_logger_id = logger.add(LOG_PATH / previous_logger_name, format="{time} {level} {message}", level="DEBUG")
            logger_stack_id.pop()
            logger_stack_id.append(previous_logger_id)


def main():
    logger.info("this should go to default log file")
    with log_to_file(f"experiment.log"):
        logger.info("this should go to experiment log file")
        for sth in ["one", "two", "three"]:
            with log_to_file(f"{sth}.log"):
                logger.info(f"{sth} will be logged to the file.")
                print(f"Print statement for {sth}")
        logger.info("this should also go to experiment log file")
    logger.info("this should also go to default log file")
    raise Exception("This should be caught and logged to default log file")


# Example usage
if __name__ == "__main__":
    main()
