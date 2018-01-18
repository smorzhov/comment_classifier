"""
It trains model

Usage: python3 train.py [-h]
"""
import argparse
from os import path
from utils import PROCESSED_DATA_PATH, get_timestamp, get_test_train_data
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


def main():
    """Main function"""
    parser = init_argparse()
    args = parser.parse_args()

    try:
        (x_train, y_train), (x_test, y_test) = get_test_train_data(args.train)
    except IOError as err:
        print(err)
        return
    top_words = 5000
    embedding_vector_length = 32
    model = get_model(
        args.model,
        top_words=top_words,
        embedding_vector_length=embedding_vector_length)
    print(model.summary())
    model.fit(x_train, y_train, epochs=10, batch_size=64)
    # Saving architecture + weights + optimizer state
    model.save('{}_{}.h5'.format(args.model, get_timestamp()))
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    main()
