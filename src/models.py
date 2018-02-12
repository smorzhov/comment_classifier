"""
Model dispatcher
"""
import keras.backend.tensorflow_backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, \
                         CuDNNGRU, Input, MaxPooling2D, concatenate
from keras.layers.core import Reshape, Flatten
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from utils import WORD2VEC_MODEL_PATH

# Dimension of word2vec
EMBEDDING_DIM = 300


class IntervalEvaluation(Callback):  # pylint: disable=R0903
    """It computes ROC AUC metrics"""

    def __init__(self, validation_data=()):
        super(Callback, self).__init__()  # pylint: disable=E1003

        self.x_val, self.y_val = validation_data
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        """It will count RIC AUC score at the end of each epoch"""
        y_pred = None
        if hasattr(self.model, 'predict_proba'):
            # for Sequentional models
            y_pred = self.model.predict_proba(self.x_val, verbose=0)
        else:
            # for models that was created using functional API
            y_pred = self.model.predict(self.x_val, verbose=0)
        self.aucs.append(roc_auc_score(self.y_val, y_pred))
        print('\repoch: {:d} - ROC AUC: {:.6f}'.format(epoch + 1,
                                                       self.aucs[-1]))


def get_model(model, gpu=1, **kwargs):
    """
    Returns model compiled keras model ready for training
    """
    with K.tf.device('/gpu:{}'.format(gpu)):
        rest = {}
        if 'use_pretrained' in kwargs:
            rest['use_pretrained'] = kwargs['use_pretrained']
        if model == 'cnn':
            if 'num_filters' in kwargs:
                rest['num_filters'] = kwargs['num_filters']
            if 'filter_sizes' in kwargs:
                rest['filter_sizes'] = kwargs['filter_sizes']
            if 'drop' in kwargs:
                rest['drop'] = kwargs['drop']
            return cnn(
                top_words=kwargs['top_words'],
                word_index=kwargs['word_index'],
                sequence_length=kwargs['sequence_length'],
                **rest)
        if model == 'lstm_cnn':
            return lstm_cnn(
                top_words=kwargs['top_words'],
                word_index=kwargs['word_index'],
                **rest)
        if model == 'gru':
            return gru(
                top_words=kwargs['top_words'],
                word_index=kwargs['word_index'],
                **rest)
        raise ValueError('Wrong model value!')


def cnn(top_words,
        sequence_length,
        word_index,
        use_pretrained=True,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        drop=0.5):
    """
    Returns compiled keras cnn model ready for training

    Best with epochs=20 (with EarlyStopping), batch_size=1024
    (ROC AUC: 0.947 - validation, ? - Kaggle).
    Training on single GPU < 20 minutes

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    """
    inputs = Input(shape=(sequence_length, ))
    embedding = get_pretrained_embedding(
        top_words, word_index)(inputs) if use_pretrained else Embedding(
            top_words, EMBEDDING_DIM)(inputs)
    reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedding)

    conv_0 = Conv2D(
        num_filters, (filter_sizes[0], EMBEDDING_DIM),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(
        num_filters, (filter_sizes[1], EMBEDDING_DIM),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(
        num_filters, (filter_sizes[2], EMBEDDING_DIM),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D(
        (sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
    maxpool_1 = MaxPooling2D(
        (sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
    maxpool_2 = MaxPooling2D(
        (sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten()(merged_tensor)
    reshape = Reshape((3 * num_filters, ))(flatten)
    dropout = Dropout(drop)(flatten)
    output = Dense(
        units=6,
        activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.01))(dropout)

    # this creates a model that includes
    model = Model(inputs, output)
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def lstm_cnn(top_words, word_index, use_pretrained=True):
    """
    Returns compiled keras lstm_cnn model ready for training

    Best with epochs=3, batch_size=256
    (ROC AUC: 0.9785 - validation, ? - Kaggle).
    Training on single GPU - 1 hours

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    """
    model = Sequential()
    if use_pretrained:
        model.add(get_pretrained_embedding(top_words, word_index))
    else:
        model.add(Embedding(top_words, EMBEDDING_DIM))
    model.add(
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(
        Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.1)))
    model.add(Dropout(0.3))
    model.add(GlobalMaxPool1D())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def gru(top_words, word_index, use_pretrained=True):
    """
    Returns compiled keras gru model ready for training

    Best with epochs=3, batch_size=256
    (ROC AUC: 0.987002 - validation, 0.9713 - Kaggle).
    Training on single GPU < 20 min.

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    """
    model = Sequential()
    if use_pretrained:
        model.add(get_pretrained_embedding(top_words, word_index))
    else:
        model.add(Embedding(top_words, EMBEDDING_DIM))
    model.add(Bidirectional(CuDNNGRU(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNGRU(64, return_sequences=False)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(clipvalue=1, clipnorm=1),
        metrics=['accuracy'])
    return model


def get_pretrained_embedding(top_words, word_index):
    """
    Returns Embedding layer with pretrained word2vec weights
    """
    word_vectors = KeyedVectors.load_word2vec_format(
        WORD2VEC_MODEL_PATH, binary=True)

    vocabulary_size = min(len(word_index) + 1, top_words)
    embedding_matrix = np.zeros((top_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= top_words:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25),
                                                   EMBEDDING_DIM)

    return Embedding(
        top_words, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
