"""
Model dispatcher
"""
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, \
                         CuDNNGRU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from utils import WORD2VEC_MODEL_PATH


class IntervalEvaluation(Callback):  # pylint: disable=R0903
    """It computes ROC AUC metrics"""

    def __init__(self, validation_data=()):
        super(Callback, self).__init__()  # pylint: disable=E1003

        self.x_val, self.y_val = validation_data
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        """It will count RIC AUC score at the end of each epoch"""
        y_pred = self.model.predict_proba(self.x_val, verbose=0)
        self.aucs.append(roc_auc_score(self.y_val, y_pred))
        print('\repoch: {:d} - ROC AUC: {:.6f}'.format(epoch + 1,
                                                       self.aucs[-1]))


def get_model(model, gpu=1, **kwargs):
    """
    Returns model compiled keras model ready for training
    """
    with K.tf.device('/gpu:{}'.format(gpu)):
        if model == 'lstm_cnn':
            return lstm_cnn(
                top_words=kwargs['top_words'],
                embedding_vector_length=kwargs['embedding_vector_length'],
                word_index=kwargs['word_index'])
        if model == 'gru':
            return gru(
                top_words=kwargs['top_words'],
                embedding_vector_length=kwargs['embedding_vector_length'],
                word_index=kwargs['word_index'])
        raise ValueError('Wrong model value!')


def lstm_cnn(top_words,
             embedding_vector_length,
             word_index,
             use_pretrained=True):
    """
    Returns compiled keras lstm_cnn model ready for training

    Best with epochs=3, batch_size=64 (0.07 loss). Training - 3 hours

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    - embedding_vector_length
    """
    model = Sequential()
    if use_pretrained:
        model.add(get_pretrained_embedding(top_words, word_index))
    else:
        model.add(Embedding(top_words, embedding_vector_length))
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


def gru(top_words, embedding_vector_length, word_index, use_pretrained=True):
    """
    Returns compiled keras gru model ready for training

    Best with epochs=3, batch_size=256 (0.9713 Kaggle ROC AUC).
    Training on single GPU < 20 min.

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    - embedding_vector_length
    """
    model = Sequential()
    if use_pretrained:
        model.add(get_pretrained_embedding(top_words, word_index))
    else:
        model.add(Embedding(top_words, embedding_vector_length))
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
    embedding_dim = 300
    word_vectors = KeyedVectors.load_word2vec_format(
        WORD2VEC_MODEL_PATH, binary=True)

    vocabulary_size = min(len(word_index) + 1, top_words)
    embedding_matrix = np.zeros((top_words, embedding_dim))
    for word, i in word_index.items():
        if i >= top_words:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25),
                                                   embedding_dim)

    return Embedding(
        top_words, embedding_dim, weights=[embedding_matrix], trainable=True)
