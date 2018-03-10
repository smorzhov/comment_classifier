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
from sklearn.metrics import confusion_matrix
import matplotlib
# generates images without having a window appear
matplotlib.use('Agg')
import matplotlib.pylab as plt
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


def load_train_data(train_file, load_augmented_train_data=False):
    """
    Loads train data
    """
    train_data = pd.read_csv(train_file, dtype={'comment_text': str}).dropna()
    train_data = train_data.drop(
        train_data[train_data['comment_text'].str.len() < 4].index)
    if load_augmented_train_data:
        corpus = [train_data]
        prefix = path.splitext(train_file)[0]
        for suffix in AUGMENTED_TRAIN_FILES:
            data = pd.read_csv(
                prefix + suffix, dtype={'comment_text': str}).dropna()
            data = data.drop(data[data['comment_text'].str.len() < 4].index)
            corpus.append(data)
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


def plot_loss_acc(history, aucs, model_path=None):
    """
    Saves into files accuracy and loss plots
    """
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path.join(model_path, 'accuracy.png'))
    plt.gcf().clear()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(aucs)
    plt.title('model loss, ROC AUC')
    plt.ylabel('loss, ROC AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'ROC AUC'], loc='upper left')
    plt.savefig(path.join(model_path, 'loss.png'))


def plot_confusion_matrices(val_labels, val_predictions, model_path=None):
    """
    Saves confusion matrices into file
    """
    matrices = []
    class_names = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]
    # transform possibilities to labels of classes
    val_preds = np.zeros(val_predictions.shape, dtype=int)
    val_preds[val_predictions > 0.5] = 1
    # build set of confusion matrices
    for cl in range(val_labels.shape[1]):
        y_true = val_labels[:, cl].tolist()
        y_pred = val_preds[:, cl].tolist()
        matrices.append(confusion_matrix(y_true, y_pred))
    for idx in range(len(matrices)):
        cur_matr = matrices[idx]
        _, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(cur_matr, cmap=plt.cm.Blues, alpha=0.2)
        plt.title(class_names[idx])
        plt.ylabel('real class', fontsize=16)
        plt.xlabel('predicted class', fontsize=16)
        # fill the plot of data from confusion matrix
        for i in range(cur_matr.shape[0]):
            for j in range(cur_matr.shape[1]):
                ax.text(
                    x=j,
                    y=i,
                    s=cur_matr[i, j],
                    va='center',
                    ha='center',
                    fontsize=24)
        file_name = '{}_confusion_matrix.png'.format(str(class_names[idx]))
        plt.savefig(path.join(model_path, file_name))
        plt.clf()
