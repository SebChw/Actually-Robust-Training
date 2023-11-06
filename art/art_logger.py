from contextlib import contextmanager
import sys
from loguru import logger
from pathlib import Path
from art.paths import LOG_PATH
logger.remove()
logger.add(sys.stdout, format="{message}", level="DEBUG", filter= lambda record: ('halt_exception' not in record['extra']) or (not record['extra']['halt_exception']))


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


logger_level = 0
already_contextualized_logers = set()


@contextmanager
def logger_contextualize(folder_path: Path, supress_stdout=False):
    if str(folder_path) in already_contextualized_logers:
        yield
        return
    already_contextualized_logers.add(str(folder_path))
    global logger_level
    logger_level+=1
    logger_id = logger.add(folder_path/"logs.log", format="{time} {level} {message}", level="DEBUG", filter= lambda record: record["extra"]['file'] == str(folder_path))
    old_sysout = sys.stdout
    sys.stdout = ArtLogInterceptor(folder_path/"stdout.log", sys.stdout, supress_stdout)
    try:
        with logger.contextualize(file=str(folder_path)):
            try:
                yield
            except Exception as e:
                logger.bind(file=str(folder_path), halt_exception=(logger_level!=1)).exception(e)
                raise e
    finally:
        logger.remove(logger_id)
        sys.stdout = old_sysout
        logger_level-=1
        already_contextualized_logers.remove(str(folder_path))


art_logger = logger

if __name__ == "__main__":
    art_logger.add(LOG_PATH/"a.txt", format="{time} {level} {message}", level="DEBUG", filter= lambda record: record["extra"]['file'] == str(LOG_PATH/"a.txt"))
    with logger_contextualize(Path("exp_logs"), supress_stdout=False):
        art_logger.info("This should go to experiment")
        print("experiment")
        logger.add(LOG_PATH/"b.txt", format="{time} {level} {message}", level="DEBUG", filter= lambda record: record["extra"]['file'] == str(LOG_PATH/"b.txt"))
        with logger_contextualize(Path("step_logs")):
            print("step")
            art_logger.info("This should go to step")
            raise Exception("B exception")
        art_logger.info("This should go to experiment again")
        print("nexperiment")
