"""
It trains model

Usage: python3 train.py [-h]
"""
import argparse
from os import path
import pandas as pd
from utils import PROCESSED_DATA_PATH, MODELS_PATH
from utils import data_to_sequence, try_makedirs
from models import get_model


def init_argparse():
    """Initializes argparse"""
    parser = argparse.ArgumentParser(
        description='Trains toxic comment classifier')
    parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help='model architecture',
        default='lstm_cnn',
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
    return parser


def plot(history, model_path=None):
    """It saves into files accuracy and loss plots"""
    import matplotlib
    # generates images without having a window appear
    matplotlib.use('Agg')
    import matplotlib.pylab as plt

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path.join(model_path, 'accuracy.png'))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path.join(model_path, 'loss.png'))


def main():
    """Main function"""
    parser = init_argparse()
    args = parser.parse_args()

    if not path.isfile(args.train):
        print('Cannot open {} file'.format(args.train))
        return
    print('Loading train and test data')
    top_words = 10000
    max_comment_length = 1000
    (x_train, y_train), (x_test, y_test) = data_to_sequence(
        args.train, top_words, max_comment_length, train_test_ratio=1.0)
    embedding_vector_length = 32
    # loading the model
    model = get_model(
        args.model,
        top_words=top_words,
        embedding_vector_length=embedding_vector_length)
    print('Training of model')
    print(model.summary())
    history = model.fit(
        x_train, y_train, validation_split=0.3, epochs=3, batch_size=64)
    # history of training
    print(history.history.keys())
    # Saving architecture + weights + optimizer state
    model_path = path.join(MODELS_PATH, '{}_{:.4f}_{:.4f}'.format(
        args.model, history.history['val_loss'][-1]
        if 'val_loss' in history.history else history.history['loss'][-1],
        history.history['val_acc'][-1]
        if 'val_acc' in history.history else history.history['acc'][-1]))
    try_makedirs(model_path)
    model.save(path.join(model_path, 'model.h5'))
    plot(history, model_path)
    # Calculate metrics of the model
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Loss: %.2f%%" % (scores[0] * 100))
    # print("Accuracy: %.2f%%" % (scores[1] * 100))

    (test_data, _), _ = data_to_sequence(
        args.test,
        top_words,
        max_comment_length,
        try_load_pickled_tokenizer=False,
        load_lables=False,
        train_test_ratio=1.0)
    print('Generating predictions')
    predictions = model.predict(test_data, batch_size=64)
    pd_predictions = pd.DataFrame({
        'id': pd.read_csv(args.test)['id'],
        'toxic': predictions[:, 0],
        'severe_toxic': predictions[:, 1],
        'obscene': predictions[:, 2],
        'threat': predictions[:, 3],
        'insult': predictions[:, 4],
        'identity_hate': predictions[:, 5]
    })
    # Rounding makes predictions much worse!
    # pd_predictions = pd_predictions.round(2)
    pd_predictions.to_csv(path.join(model_path, 'predictions.csv'), index=False)


if __name__ == '__main__':
    main()
