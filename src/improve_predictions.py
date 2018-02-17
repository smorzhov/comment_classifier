"""
It tryies to improve predictions, but now it makes predictions much worse

Usage: python improve_predictions.py [-h]
"""
from argparse import ArgumentParser
from os import path
import pandas as pd

THRESHOLD = 0.01


def init_argparse():
    """Initializes argparse"""
    parser = ArgumentParser(description='Trains toxic comment classifier')
    parser.add_argument(
        '-p',
        '--predictions',
        nargs='?',
        help='path to predictions.csv file',
        type=str)
    return parser


def transform(x):
    """Transforms value x"""
    if x < THRESHOLD:
        return 0.0
    elif x > 1 - THRESHOLD:
        return 1.0
    else:
        return x


def main():
    """Main function"""
    args = init_argparse().parse_args()
    result_file = path.splitext(args.predictions)[0] + '_trans.csv'
    columns = [
        'identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic'
    ]
    predictions = pd.read_csv(args.predictions, encoding='utf-8')
    predictions[columns] = predictions[columns].applymap(transform)
    predictions.to_csv(result_file, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
