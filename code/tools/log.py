import logging
import sys


def setup_logging(log_path=None, log_level=logging.INFO):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(name)s:%(lineno)d %(levelname)s :: %(message)s')

    # Create console handler to write to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        # Create file handler, attach formatter and add it to the logger
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
