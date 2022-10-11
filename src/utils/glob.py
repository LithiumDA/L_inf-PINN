import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import os


logger = None
summary_writer = None
config = None

def setup_logging():
    global logger
    if logger is not None:
        return logger
    """Enable pretty logging and sets the level to DEBUG."""
    logging.addLevelName(logging.DEBUG, "D")
    logging.addLevelName(logging.INFO, "I")
    logging.addLevelName(logging.WARNING, "W")
    logging.addLevelName(logging.ERROR, "E")
    logging.addLevelName(logging.CRITICAL, "C")

    formatter = logging.Formatter(
        fmt=("%(levelname)s%(asctime)s" " [%(module)s:%(lineno)d] %(message)s"),
        datefmt="%m%d %H:%M:%S",
    )

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("run.log")
    file_handler.setFormatter(formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def setup_board():
    global summary_writer
    if summary_writer is None:
        summary_writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), 'tensorboard'))
    return summary_writer

def setup_cfg(cfg):
    global config
    config = cfg
