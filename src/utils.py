"""
Some useful utilities
"""
import logging
import datetime
try:
    import cPickle as pickle
except ImportError:
    import pickle
from os import path, makedirs
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
"""
Absolute utils.py file path. It is considered as the project root path.
"""
CWD = path.dirname(path.realpath(__file__))
"""
It must contain files with raw data
"""
RAW_DATA_PATH = path.join(CWD, 'data', 'raw')
WORD2VEC_MODEL_PATH = path.join(RAW_DATA_PATH,
                                'GoogleNews-vectors-negative300.bin')
GLOVE_6B_MODEL_PATH = path.join(RAW_DATA_PATH, 'glove.6B.300d.txt')
GLOVE_840B_MODEL_PATH = path.join(RAW_DATA_PATH, 'glove.840B.300d.txt')
FAST_TEXT_MODEL_PATH = path.join(RAW_DATA_PATH, 'crawl-300d-2M.vec')
"""
Processed test, train and other files used in training (testing) process must be saved here.
By default, this directory is being ignored by GIT. It is not recommended
to exclude this directory from .gitignore unless there is no extreme necessity.
"""
PROCESSED_DATA_PATH = path.join(CWD, 'data', 'processed')
LOG_PATH = path.join(CWD, 'log')
"""
Trained models must be stored here
"""
MODELS_PATH = path.join(CWD, 'models')
"""
Pickled objects must be stored here
"""
PICKLES_PATH = path.join(CWD, 'pickles')
STOP_WORDS_PATH = path.join(RAW_DATA_PATH, 'stopwords.txt')
AUGMENTED_TRAIN_FILES = ['_de.csv', '_fr.csv', '_es.csv']


def get_stop_words(stop_words=STOP_WORDS_PATH):
    """
    Params:
    - stop_words - path to file with stop words.
    Each line of the file must contains one word

    Returns set with stop words
    """
    return set(open(stop_words).read().split())


def try_makedirs(name):
    """
    Makes path if it doesn't exist
    """
    try:
        if not path.exists(name):
            # Strange, but it may raise winerror 123
            makedirs(name)
    except OSError:
        return


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


def get_timestamp():
    """
    Returns timestamp in YYYY-MM-DDTHH:MM:SS format
    """
    return datetime.datetime.today().strftime('%Y-%m-%dT%H:%M:%S')


def load_train_data(train_file, load_augmented_train_data=False):
    """
    Loads train data
    """
    train_data = pd.read_csv(train_file, dtype={'comment_text': str})
    if load_augmented_train_data:
        corpus = [train_data]
        prefix = path.splitext(train_file)[0]
        for suffix in AUGMENTED_TRAIN_FILES:
            corpus.append(
                pd.read_csv(prefix + suffix, dtype={'comment_text': str}))
        # merge augmented and raw train data
        train_data = pd.concat(corpus, ignore_index=True)
    return train_data


def load_test_train_data(train_file,
                         test_file,
                         load_augmented_train_data=False,
                         num_words=None,
                         max_comment_length=500):
    """
    Returns test typle and train list
    """
    train_data = load_train_data(train_file, load_augmented_train_data)
    test_data = pd.read_csv(test_file, dtype={'comment_text': str})
    tokenizer = Tokenizer(num_words, oov_token='unk')
    tokenizer.fit_on_texts(
        pd.concat(
            [train_data['comment_text'], test_data['comment_text']],
            ignore_index=True))
    # Pull train, test data and their labels
    x_train = []
    y_train = []
    for seq, row in zip(
            tokenizer.texts_to_sequences_generator(train_data['comment_text']),
            train_data[[
                'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                'identity_hate'
            ]].iterrows()):
        x_train.append(seq)
        y_train.append(row[1].values.tolist())
    # Truncate and pad input sequences
    x_train = sequence.pad_sequences(x_train, maxlen=max_comment_length)
    x_test = sequence.pad_sequences(
        tokenizer.texts_to_sequences(test_data['comment_text']),
        maxlen=max_comment_length)
    return (np.asarray(x_train),
            np.asarray(y_train)), np.asarray(x_test), tokenizer.word_index
