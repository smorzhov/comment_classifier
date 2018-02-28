"""
Model dispatcher
"""
from os import path
try:
    import cPickle as pickle
except ImportError:
    import pickle
import keras.backend.tensorflow_backend as K
from keras import regularizers
from keras import initializers
from keras import constraints
from keras.models import Sequential, Model
from keras.layers import Dense, CuDNNLSTM, Bidirectional, Dropout, PReLU, \
                         CuDNNGRU, MaxPooling2D, Input, Activation, \
                         SpatialDropout1D, GlobalAveragePooling1D, \
                         GlobalMaxPooling1D, concatenate
from keras.layers.core import Reshape, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from keras.utils import multi_gpu_model
from sklearn.metrics import roc_auc_score
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from utils import WORD2VEC_MODEL_PATH, GLOVE_6B_MODEL_PATH, \
                  GLOVE_840B_MODEL_PATH, FAST_TEXT_MODEL_PATH, PICKLES_PATH, \
                  try_makedirs

EMBEDDING_DIM = 300


class IntervalEvaluation(Callback):  # pylint: disable=R0903
    """Computes ROC AUC metrics"""

    def __init__(self, validation_data=()):
        super(Callback, self).__init__()  # pylint: disable=E1003

        self.x_val, self.y_val = validation_data
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Count ROC AUC score at the end of each epoch
        """
        y_pred = None
        if hasattr(self.model, 'predict_proba'):
            # for Sequentional models
            y_pred = self.model.predict_proba(self.x_val, verbose=0)
        else:
            # for models that was created using functional API
            y_pred = self.model.predict(self.x_val, verbose=0)
        self.aucs.append(roc_auc_score(self.y_val, y_pred))
        print(
            '\repoch: {:d} - ROC AUC: {:.6f}'.format(epoch + 1, self.aucs[-1]))


def get_gpus(gpus):
    """
    Returns a list of integers (numbers of gpus)
    """
    return list(map(int, gpus.split(',')))


def get_model(model, gpus=1, **kwargs):
    """
    Returns compiled keras parallel model ready for training
    and base model that must be used for saving weights

    Params:
    - model - model type
    - gpus - a list with numbers of GPUs
    """
    rest = {'sequence_length': kwargs['sequence_length']}
    if 'pretrained' in kwargs:
        rest['pretrained'] = kwargs['pretrained']
    if model == 'cnn':
        if 'num_filters' in kwargs:
            rest['num_filters'] = kwargs['num_filters']
        if 'filter_sizes' in kwargs:
            rest['filter_sizes'] = kwargs['filter_sizes']
        if 'drop' in kwargs:
            rest['drop'] = kwargs['drop']
        return cnn(
            gpus=gpus,
            top_words=kwargs['top_words'],
            word_index=kwargs['word_index'],
            **rest)
    if model == 'lstm':
        return lstm(
            gpus=gpus,
            top_words=kwargs['top_words'],
            word_index=kwargs['word_index'],
            **rest)
    if model == 'gru':
        return gru(
            gpus=gpus,
            top_words=kwargs['top_words'],
            word_index=kwargs['word_index'],
            **rest)
    raise ValueError('Wrong model value!')


def cnn(top_words,
        sequence_length,
        word_index,
        gpus,
        pretrained=None,
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
    inputs = Input(shape=(sequence_length,))
    embedding = get_pretrained_embedding(top_words, sequence_length, word_index,
                                         pretrained)(inputs)
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
    reshape = Reshape((3 * num_filters,))(flatten)
    dropout = Dropout(drop)(reshape)
    output = Dense(
        units=6, activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.01))(dropout)

    gpus = get_gpus(gpus)
    if len(gpus) == 1:
        with K.tf.device('/gpu:{}'.format(gpus[0])):
            model = Model(inputs, output)
            parallel_model = model
    else:
        with K.tf.device('/cpu:0'):
            # creates a model that includes
            model = Model(inputs, output)
        parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return parallel_model, model


def lstm(top_words, sequence_length, word_index, gpus, pretrained=None):
    """get_pretrained_embedding(top_words, sequence_length, word_index, pretrained)
    Returns compiled keras lstm model ready for training

    Best with epochs=3, batch_size=256
    (ROC AUC: 0.9785 - validation, ? - Kaggle).
    Training on single GPU - 1 hours

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    - pretrained - None, 'word2vec', 'glove6B', 'glove840B', 'fasttext'
    """
    units = 100
    inputs = Input(shape=(sequence_length,), dtype='int32')
    x = get_pretrained_embedding(top_words, sequence_length, word_index,
                                 pretrained)(inputs)
    # For mor detais about kernel_constraint - see chapter 5.1
    # in http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    x = Bidirectional(
        CuDNNLSTM(
            units,
            recurrent_regularizer=regularizers.l2(),
            return_sequences=True),
        merge_mode='concat')(x)
    x = Activation('tanh')(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(
        CuDNNLSTM(
            units,
            recurrent_regularizer=regularizers.l2(),
            return_sequences=False),
        merge_mode='concat')(x)
    x = Activation('tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(6)(x)
    output = PReLU()(x)
    gpus = get_gpus(gpus)
    if len(gpus) == 1:
        with K.tf.device('/gpu:{}'.format(gpus[0])):
            model = Model(inputs, output)
            parallel_model = model
    else:
        with K.tf.device('/cpu:0'):
            # creates a model that includes
            model = Model(inputs, output)
        parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(clipvalue=1, clipnorm=1),
        metrics=['accuracy'])
    return parallel_model, model


def gru(top_words, sequence_length, gpus, word_index, pretrained=None):
    """
    Returns compiled keras gru model ready for training

    Best with epochs=3, batch_size=256
    (ROC AUC: 0.987002 - validation, 0.9713 - Kaggle).
    Training 2 GPUs < 20 min.

    Params:
    - top_words - load the dataset but only keep the top n words, zero the rest
    - pretrained - None, 'word2vec', 'glove6B', 'glove840B', 'fasttext'
    """
    # units = 2 * EMBEDDING_DIM
    units = 300
    inputs = Input(shape=(sequence_length,))
    x = get_pretrained_embedding(top_words, sequence_length, word_index,
                                 pretrained)(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(
        CuDNNGRU(
            units,
            kernel_initializer=initializers.he_normal(),
            recurrent_regularizer=regularizers.l2(),
            return_sequences=True),
        merge_mode='concat')(x)
    x = Dropout(0.5)(x)
    x = PReLU()(x)
    x = Bidirectional(
        CuDNNGRU(
            units,
            kernel_initializer=initializers.he_normal(),
            recurrent_regularizer=regularizers.l2(),
            return_sequences=True),
        merge_mode='concat')(x)
    avg_pool_1 = GlobalAveragePooling1D()(x)
    max_pool_1 = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool_1, max_pool_1])
    outputs = Dense(6, activation='sigmoid')(conc)

    gpus = get_gpus(gpus)
    if len(gpus) == 1:
        with K.tf.device('/gpu:{}'.format(gpus[0])):
            model = Model(inputs, outputs)
            parallel_model = model
    else:
        with K.tf.device('/cpu:0'):
            # creates a model that includes
            model = Model(inputs, outputs)
        parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(clipvalue=1, clipnorm=1),
        metrics=['accuracy'])
    return parallel_model, model


def get_pretrained_embedding(top_words, sequence_length, word_index,
                             pretrained):
    """
    Returns Embedding layer with pretrained word2vec weights

    Params:
    - pretrained - None, 'word2vec', 'glove6B', 'glove840B', 'fasttext'
    """
    word_vectors = {}
    if pretrained == 'word2vec':
        word_vectors = KeyedVectors.load_word2vec_format(
            WORD2VEC_MODEL_PATH, binary=True)
    elif pretrained == 'glove6B':
        word_vectors = load_txt_model(GLOVE_6B_MODEL_PATH)
    elif pretrained == 'glove840B':
        word_vectors = load_txt_model(GLOVE_840B_MODEL_PATH)
    elif pretrained == 'fasttext':
        word_vectors = load_txt_model(FAST_TEXT_MODEL_PATH)
    else:
        return Embedding(
            input_dim=top_words,
            output_dim=EMBEDDING_DIM,
            input_length=sequence_length,
            trainable=False,
            mask_zero=False)

    embedding_matrix = np.zeros((top_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= top_words:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0,
                                                   np.sqrt(0.25), EMBEDDING_DIM)

    return Embedding(
        input_dim=top_words,
        output_dim=EMBEDDING_DIM,
        input_length=sequence_length,
        weights=[embedding_matrix],
        trainable=False,
        mask_zero=False)


def load_txt_model(model_path):
    """
    Returns pretrained serialized model saved in text format
    where numbers are separated with spaces
    """
    try_makedirs(PICKLES_PATH)
    pickled_model = path.join(PICKLES_PATH,
                              '{}.pickle'.format(path.basename(model_path)))
    try:
        # load ready text model
        with open(pickled_model, 'rb') as model:
            return pickle.load(model)
    except:
        # form text model
        with open(model_path, 'r') as file:
            model = {}
            for line in file:
                splitLine = line.split()
                # pull word
                word = splitLine[0]
                # pull features
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            with open(pickled_model, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return model
