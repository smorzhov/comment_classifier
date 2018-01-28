"""
Model dispatcher
"""
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import CuDNNGRU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop


def get_model(model, gpu=1, **kwargs):
    """
    Returns model compiled keras model ready for training
    """
    with K.tf.device('/gpu:{}'.format(gpu)):
        if model == 'lstm_cnn':
            return lstm_cnn(**kwargs)
        if model == 'gru':
            return gru(**kwargs)
        raise ValueError('Wrong model value!')


def lstm_cnn(top_words, embedding_vector_length, **kwargs):
    """
    Returns compiled keras lstm_cnn model ready for training

    Best with epochs=3, batch_size=64 (0.07 loss). Training - 3 hours

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    - embedding_vector_length
    - **kwargs - keras specific Embedding() arguments
    """
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, **kwargs))
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


def gru(top_words, embedding_vector_length, **kwargs):
    """
    Returns compiled keras gru model ready for training

    Best with epochs=2, batch_size=256 (0.068 loss). Training < 30 min

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    - embedding_vector_length
    - **kwargs - keras specific Embedding() arguments
    """
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, **kwargs))
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
