"""
Model dispatcher
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding


def get_model(model, **kwargs):
    """
    Returns model compiled keras model ready for training
    """
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
    model.add(LSTM(100))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
