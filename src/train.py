"""
Trains model

Usage: python train.py [-h]
"""
from time import clock
from argparse import ArgumentParser
from os import path, environ
import pandas as pd
import numpy as np
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from utils import (PROCESSED_DATA_PATH, MODELS_PATH, load_test_train_data,
                   try_makedirs, plot_loss_acc, plot_confusion_matrices)
from models import get_model, IntervalEvaluation

# False - don't use augmented train data, True - use it
TRAIN_PARAMS = {
    'cnn': {
        False: {
            'epochs': 20,
            'batch_size': 1024,
            'pretrained': 'glove840B'  # 'word2vec', 'glove6B', 'glove840B'
        },
        True: {
            'epochs': 20,
            'batch_size': 1024,
            'pretrained': 'glove840B'  # 'word2vec', 'glove6B', 'glove840B'
        }
    },
    'lstm': {
        False: {
            'epochs': 5,
            'batch_size': 256,
            'pretrained': 'glove840B'
        },
        True: {
            'epochs': 10,
            'batch_size': 256,
            'pretrained': 'glove840B'
        }
    },
    'gru': {
        False: {
            'epochs': 5,
            'batch_size': 256,
            'pretrained': 'glove840B'
        },
        True: {
            'epochs': 9,
            'batch_size': 5 * 256,
            'pretrained': 'glove840B'
        }
    }
}


def init_argparse():
    """
    Initializes argparse

    Returns parser
    """
    parser = ArgumentParser(description='Trains toxic comment classifier')
    parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help='model architecture (lstm, gru, cnn)',
        default='gru',
        type=str)
    parser.add_argument(
        '-t',
        '--train',
        nargs='?',
        help='path to train.csv file',
        default=path.join(PROCESSED_DATA_PATH, 'train.csv'),
        type=str)
    parser.add_argument(
        '-T',
        '--test',
        nargs='?',
        help='path to test.csv file',
        default=path.join(PROCESSED_DATA_PATH, 'test.csv'),
        type=str)
    parser.add_argument(
        '--load_augmented',
        help='Use augmente data for training',
        action='store_true')
    parser.add_argument(
        '--gpus',
        nargs='?',
        help="A list of GPU device numbers ('1', '1,2,5')",
        default=0,
        type=str)
    parser.add_argument(
        '--cv', help="Cross validation of the model", action='store_true')
    return parser


def train_and_predict(data, labels, test_data, word_index, top_words, args):
    """
    Trains model and makes predictions file
    """
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.1, random_state=42)
    # loading the model
    parallel_model, model = get_model(
        args.model,
        gpus=args.gpus,
        top_words=top_words,
        word_index=word_index,
        pretrained=TRAIN_PARAMS[args.model][args.load_augmented]['pretrained'],
        sequence_length=train_data.shape[1])
    # sequence_length = train_data.shape[1]= max_comment_length
    print('Training model')
    print(model.summary())
    ival = IntervalEvaluation(validation_data=(val_data, val_labels))
    history = parallel_model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=TRAIN_PARAMS[args.model][args.load_augmented]['epochs'],
        batch_size=TRAIN_PARAMS[args.model][args.load_augmented]['batch_size'],
        callbacks=[
            ival,
            EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        ])
    # history of training
    # print(history.history.keys())
    # Saving architecture + weights + optimizer state
    model_path = path.join(MODELS_PATH, '{}_{:.4f}_{:.4f}_{:.4f}'.format(
        args.model, ival.aucs[-1], history.history['val_loss'][-1]
        if 'val_loss' in history.history else history.history['loss'][-1],
        history.history['val_acc'][-1]
        if 'val_acc' in history.history else history.history['acc'][-1]))
    try_makedirs(model_path)
    plot_model(model, path.join(model_path, 'model.png'), show_shapes=True)
    plot_loss_acc(history, ival.aucs, model_path)
    print('Saving model')
    model.save(path.join(model_path, 'model.h5'))
    # Building confusion matrices for every class for validation data
    print("Building confusion matrices")
    val_preds = model.predict(
        val_data,
        batch_size=TRAIN_PARAMS[args.model][args.load_augmented]['batch_size'])
    plot_confusion_matrices(val_labels, val_preds, model_path)

    print('Generating predictions')
    predictions = model.predict(
        test_data,
        batch_size=TRAIN_PARAMS[args.model][args.load_augmented]['batch_size'])
    pd.DataFrame({
        'id': pd.read_csv(args.test)['id'],
        'toxic': predictions[:, 0],
        'severe_toxic': predictions[:, 1],
        'obscene': predictions[:, 2],
        'threat': predictions[:, 3],
        'insult': predictions[:, 4],
        'identity_hate': predictions[:, 5]
    }).to_csv(
        path.join(model_path, 'predictions.csv'), index=False)
    # Don't round predictions.
    # Rounding makes predictions much worse!


def evaluate_model(data, labels, test_data, word_index, top_words,
                   max_comment_length, args):
    """
    Evaluates metrics by cross-validation the `model`
    """

    def print_stage(index):
        print('\n===============================')
        print('STAGE {}'.format(index))
        print('===============================\n')

    def print_statistics(stage, arr):
        print('{}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(
            stage, np.amin(arr), np.mean(arr), np.std(arr), np.max(arr)))

    start = clock()
    print('Evaluating {} model'.format(args.model))
    seed = 42
    np.random.seed(seed)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = {'loss': [], 'acc': [], 'roc': []}
    i = -1
    for train, test in kfold.split(data, labels):
        i += 1
        print_stage(i)
        # loading the model
        parallel_model, _ = get_model(
            args.model,
            gpus=args.gpus,
            top_words=top_words,
            word_index=word_index,
            pretrained=TRAIN_PARAMS[args.model][args.load_augmented][
                'pretrained'],
            sequence_length=data[train].shape[1])
        parallel_model.fit(
            data[train],
            labels[train],
            validation_data=(data[test], labels[test]),
            epochs=TRAIN_PARAMS[args.model][args.load_augmented]['epochs'],
            batch_size=TRAIN_PARAMS[args.model][args.load_augmented][
                'batch_size'])
        scores = parallel_model.evaluate(
            data[test],
            labels[test],
            batch_size=TRAIN_PARAMS[args.model][args.load_augmented][
                'batch_size'],
            verbose=0)
        # scores - ['loss', 'acc']
        predictions = parallel_model.predict(
            data[test],
            batch_size=TRAIN_PARAMS[args.model][args.load_augmented][
                'batch_size'])
        cvscores['loss'].append(scores[0])
        cvscores['acc'].append(scores[1])
        cvscores['roc'].append(roc_auc_score(labels[test], predictions))
    print('      min     mean    std     max')
    print_statistics('roc ', cvscores['roc'])
    print_statistics('loss', cvscores['loss'])
    print_statistics('acc ', cvscores['acc'])
    m, s = divmod(clock() - start, 60)
    h, m = divmod(m, 60)
    print('Spent time: {}:{}:{} (hh:mm:ss)'.format(int(h), int(m), int(s)))


def main():
    """
    Main function
    """
    args = init_argparse().parse_args()

    environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if not path.isfile(args.train):
        print('Cannot open {} file'.format(args.train))
        return
    print('Loading train and test data')
    top_words = 130000
    max_comment_length = 500
    (data, labels), test_data, word_index = load_test_train_data(
        train_file=args.train,
        test_file=args.test,
        num_words=top_words,
        load_augmented_train_data=args.load_augmented,
        max_comment_length=max_comment_length)
    if args.cv:
        evaluate_model(data, labels, test_data, word_index, top_words,
                       max_comment_length, args)
    else:
        train_and_predict(data, labels, test_data, word_index, top_words, args)


if __name__ == '__main__':
    main()
