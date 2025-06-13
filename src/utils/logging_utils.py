import logging
import sys
import os
from ..configs import main_config as config

def setup_logger(logger_name, level=None, log_file=None, add_console_handler=True, add_file_handler=True):
    if level is None:
        level = config.LOGGING_LEVEL

    logger = logging.getLogger(logger_name)

    # Prevent adding duplicate handlers if the logger already has them
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if add_console_handler:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if add_file_handler:
        if log_file is None:
            log_file = config.APPLICATION_LOG_FILE

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                print(f"CRITICAL ERROR: Could not create log directory ({log_dir}): {e}. Logging to file will not be possible.")

        if log_dir and os.path.exists(log_dir):
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            print(f"WARNING: Directory for log file ({log_dir}) does not exist. Logging to file will not be possible.")

    return logger