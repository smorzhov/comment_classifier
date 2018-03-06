"""
Trains model

Usage: python model_correlation.py [-h]
"""

from argparse import ArgumentParser
import pandas as pd
from scipy.stats import ks_2samp


def init_argparse():
    """
    Initializes argparse

    Returns parser
    """
    parser = ArgumentParser(description='Trains toxic comment classifier')
    parser.add_argument(
        '-f',
        '--first',
        nargs='?',
        help='predictions of the first model in csv format',
        type=str)
    parser.add_argument(
        '-s',
        '--second',
        nargs='?',
        help='predictions of the second model in csv format',
        type=str)
    return parser


def corr(first_file, second_file):
    """
    Prints some useful information about the correlation
    of the models predictions
    """
    # assuming first column is `class_name_id`
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    class_names = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]

    for class_name in class_names:
        # all correlations
        print('\nClass: {}'.format(class_name))
        print('Pearson\'s correlation score: {:.6f}'.format(
            first_df[class_name].corr(second_df[class_name],
                                      method='pearson')))
        print('Kendall\'s correlation score: {:.6f}'.format(
            first_df[class_name].corr(second_df[class_name],
                                      method='kendall')))
        print('Spearman\'s correlation score: {:.6f}'.format(
            first_df[class_name].corr(
                second_df[class_name], method='spearman')))
        ks_stat, p_value = ks_2samp(first_df[class_name].values,
                                    second_df[class_name].values)
        print('Kolmogorov-Smirnov test:  KS-stat = {:.6f}  p-value = {:.3e}\n'.
              format(ks_stat, p_value))


def main():
    """
    Main function
    """
    args = init_argparse().parse_args()
    corr(args.first, args.second)


if __name__ == '__main__':
    main()
