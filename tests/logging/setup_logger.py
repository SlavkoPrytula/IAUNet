import sys
sys.path.append("./")

from utils.logging import setup_logger
from configs import LOGGING_NAME
logger = setup_logger(name=LOGGING_NAME)



logger_name = 'iaunet'
log_file1 = 'tests/logging/my_log1.log'
log_file2 = 'tests/logging/my_log2.log'
logger = setup_logger(logger_name, log_files=[log_file1, log_file2])

logger.info('This is an info message with some details: lr: 6.9970e-05, eta: 0:21:13')

