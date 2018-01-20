"""
It trains model

Usage: python3 train.py [-h]
"""
import argparse
from os import path
from utils import PROCESSED_DATA_PATH, MODELS_PATH
from utils import get_timestamp, get_test_train_data, try_makedirs
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
    (x_train, y_train), (x_test, y_test) = get_test_train_data(
        args.train, top_words, 1000)
    embedding_vector_length = 32
    model = get_model(
        args.model,
        top_words=top_words,
        embedding_vector_length=embedding_vector_length)
    print('Training')
    print(model.summary())
    history = model.fit(
        x_train, y_train, validation_split=0.25, epochs=2, batch_size=64)
    print(history.history.keys())
    # Saving architecture + weights + optimizer state
    model_path = path.join(MODELS_PATH, '{}_{}'.format(args.model,
                                                       get_timestamp()))
    try_makedirs(model_path)
    model.save(path.join(model_path, 'model.h5'))
    plot(history, model_path)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Loss: %.2f%%" % (scores[0] * 100))
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    main()
