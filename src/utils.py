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
FAST_TEXT_MODEL_PATH = path.join(RAW_DATA_PATH, 'wiki.en.vec')
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


def try_makedirs(name):
    """Makes path if it doesn't exist"""
    try:
        if not path.exists(name):
            # Stragne, but it may raise winerror 123
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
    """Returns timestamp in YYYY-MM-DDTHH:MM:SS format"""
    return datetime.datetime.today().strftime('%Y-%m-%dT%H:%M:%S')


def load_test_train_data(train_file,
                         test_file,
                         num_words=None,
                         max_comment_length=500,
                         try_load_pickled_tokenizer=False):
    """Returns test typle and train list"""

    def init_tokenizer():
        """Initializes tokenizer"""
        tokenizer = Tokenizer(num_words)
        tokenizer.fit_on_texts(
            pd.concat(
                [train_data['comment_text'], test_data['comment_text']],
                ignore_index=True))
        # Tokenizer takes a lot of time to build an index.
        # That is why it is good to store it as a pickle.
        try_makedirs(PICKLES_PATH)
        file_name = '{}_tokenizer.pickle'.format(
            path.splitext(path.basename(train_file)[0]))
        with open(path.join(PICKLES_PATH, file_name), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer

    # Returns shuffled sample of DataFrame
    train_data = pd.read_csv(
        train_file, dtype={
            'comment_text': str
        }).sample(frac=1)
    test_data = pd.read_csv(test_file, dtype={'comment_text': str})
    tokenizer = None
    if try_load_pickled_tokenizer:
        try:
            with open(path.join(PICKLES_PATH, 'tokenizer.pickle'),
                      'rb') as handle:
                tokenizer = pickle.load(handle)
        except:
            tokenizer = init_tokenizer()
    else:
        tokenizer = init_tokenizer()
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
