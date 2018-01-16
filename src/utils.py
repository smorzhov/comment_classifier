"""
Some useful utilities
"""
import logging
from os import path, makedirs
"""
The variant with MODE = 'cpu' must be considered.
The app must work properly in cpu mode also!
"""
MODE = 'gpu'
"""
Absolute utils.py file path. It is considered as the project root path.
"""
CWD = path.dirname(path.realpath(__file__))
"""
It must contain files with raw data
"""
RAW_DATA_PATH = path.join(CWD, 'data', 'raw')
"""
Processed test, train and other files used in training (testing) process must be saved here.
By default, this directory is being ignored by GIT. It is not recommended
to exclude this directory from .gitignore unless there is no extreme necessity.
"""
PROCESSED_DATA_PATH = path.join(CWD, 'data', 'processed')
LOG_PATH = path.join(CWD, 'log')


def try_makedirs(name):
    """Makes path if it doesn't exist"""
    if not path.exists(name):
        makedirs(name)


def get_logger(file):
    """
    Returns logger object

    Usage:
    ```python
    logger = get_logger('path_to_log_file')
    logger.info('It will be written into <path_to_log_file> file')
    ```
    """
    log_path = path.join(CWD, 'log')
    try_makedirs(log_path)
    logging.basicConfig(
        format=u'%(levelname)-8s [%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=path.join(log_path, file))
    return logging
