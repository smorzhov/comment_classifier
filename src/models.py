"""
Model dispatcher
"""

import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding


def get_model(model, gpu=1, **kwargs):
    """
    Returns model compiled keras model ready for training
    """
    with K.tf.device('/gpu:{}'.format(gpu)):
        if model == 'lstm_cnn':
            return lstm_cnn(**kwargs)
        raise ValueError('Wrong model value!')


def lstm_cnn(top_words, embedding_vector_length, **kwargs):
    """
    Returns compiled keras lstm_cnn model ready for training

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
        Bidirectional(
            LSTM(
                100, return_sequences=True, dropout=0.1,
                recurrent_dropout=0.1)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
