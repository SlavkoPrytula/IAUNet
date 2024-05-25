import logging
from datetime import datetime
import sys
sys.path.append("./")
from configs import LOGGING_NAME


class Formatter(logging.Formatter):
    """Custom logging formatter to match the desired output."""
    def format(self, record):
        prefix = f'{datetime.now().strftime("%m/%d %H:%M:%S")} - {record.name} - {record.levelname} -'
        formatted_message = f'{prefix} {record.getMessage()}'
        
        return formatted_message


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



logger_name = 'mm'
log_file = 'tests/logging/my_log.log'
# logger = setup_logger(logger_name, None)
logger = setup_logger(logger_name)
logger = setup_logger(logger_name)

logger.info('This is an info message with some details: lr: 6.9970e-05, eta: 0:21:13')

