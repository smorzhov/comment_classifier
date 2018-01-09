"""
Some useful utilities
"""
import logging
from os import path, makedirs

MODE = 'gpu'

CWD = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(CWD, 'data')
LOG_PATH = path.join(CWD, 'log')


def try_makedirs(name):
    """Makes path if it doesn't exist"""
    if not path.exists(name):
        makedirs(name)


def get_logger(file):
    """Returns logger object"""
    log_path = path.join(CWD, 'log')
    try_makedirs(log_path)
    logging.basicConfig(
        format=u'%(levelname)-8s [%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=path.join(log_path, file))
    return logging
