"""
It trains model

Usage: python3 train.py [-h]
"""
import argparse
from os import path
from utils import PROCESSED_DATA_PATH
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
        default=path.join(PROCESSED_DATA_PATH, 'test.csv'),
        type=str)
    return parser


def main():
    """Main function"""
    parser = init_argparse()
    args = parser.parse_args()

    # TODO: load train validation data
    top_words = 5000
    embedding_vector_length = 32
    model = get_model(
        args.model,
        top_words=top_words,
        embedding_vector_length=embedding_vector_length)
    model.fit()
    # TODO: store parameters into file to be able to reproduce training (do not forget to save random seed!)
    print(model.summary())
    # TODO: training


if __name__ == '__main__':
    main()
