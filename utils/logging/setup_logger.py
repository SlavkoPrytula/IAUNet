import logging
import sys
from datetime import datetime
import functools
from configs import LOGGING_NAME


class Formatter(logging.Formatter):
    """Custom logging formatter to match the desired output."""
    def format(self, record):
        prefix = f'{datetime.now().strftime("%m/%d %H:%M:%S")} - {record.name} - {record.levelname} -'
        formatted_message = f'{prefix} {record.getMessage()}'
        
        return formatted_message


@functools.lru_cache()
def setup_logger(
        name=LOGGING_NAME, 
        log_file=None, 
        level=logging.INFO):
    """Function to set up a logger with a specific format and file handler."""

    formatter = Formatter()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.propagate = False

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
