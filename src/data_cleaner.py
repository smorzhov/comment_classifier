"""
It cleans comments

Usage: python3 data_cleaner [-h]
Also, it can be import into another module
"""
import argparse
from os import path
import pandas as pd
from utils import PROCESSED_DATA_PATH, RAW_DATA_PATH


def init_argparse():
    """Initializes argparse"""
    parser = argparse.ArgumentParser(
        description='Trains toxic comment classifier')
    parser.add_argument(
        '-r',
        '--raw',
        nargs='?',
        help='Path to raw test.csv file',
        default=path.join(RAW_DATA_PATH, 'train.csv'),
        type=str)
    parser.add_argument(
        '-p',
        '--processed',
        nargs='?',
        help='Path where clean data will be saved',
        default=PROCESSED_DATA_PATH,
        type=str)
    return parser


def clean_comment(comment):
    """
    It cleans comment

    Args:
    - comment - raw comment string

    Returns clean comment string in utf-8. Original `comment` is not transformed
    """
    raise NotImplementedError


def clean(data):
    """It cleans comments from test.csv"""
    for index, row in data.iterrows():
        data.set_value(index, 'comment_text', clean_comment(row['comment_text']))


def main():
    """Main function"""
    parser = init_argparse()
    args = parser.parse_args()

    train = pd.read_csv(args.raw)
    clean(train)
    train.to_csv(
        path.join(args.processed, path.basename(args.raw)), index=False)


if __name__ == '__main__':
    main()
