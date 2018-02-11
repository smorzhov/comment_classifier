"""
It cleans comments

Usage: python3 data_cleaner [-h]
Also, it can be imported into another module
"""
import argparse
from os import path
from functools import partial
from multiprocessing import Pool, cpu_count
import pandas as pd
import re
import copy
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from autocorrect import spell
from utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, try_makedirs
from collections import deque


def init_argparse():
    """Initializes argparse"""
    parser = argparse.ArgumentParser(
        description='Trains toxic comment classifier')
    parser.add_argument(
        '--train',
        nargs='?',
        help='Path to raw train.csv file',
        default=path.join(RAW_DATA_PATH, 'train.csv'),
        type=str)
    parser.add_argument(
        '--test',
        nargs='?',
        help='Path to raw test.csv file',
        default=path.join(RAW_DATA_PATH, 'test.csv'),
        type=str)
    parser.add_argument(
        '-p',
        '--processed',
        nargs='?',
        help='Path where clean data will be saved',
        default=PROCESSED_DATA_PATH,
        type=str)
    parser.add_argument(
        '--cpus',
        nargs='?',
        help='Number of CPUs to use. By default, it will use all CPUs',
        default=cpu_count(),
        type=int)
    return parser


def clean_spaces(comment):
    """
    This function replaces all double (triple and so on) spaces to one

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    comment = re.sub(r'\s+', ' ', comment)
    comment = comment.strip()
    return comment


def clean_punctuation_spaces(comment):
    """
    This function set space to punctuation marks ". , ? ! : ;"

    Args:
    - comment - raw comment string

    Doesn't return value
    It's change argument
    """
    # TODO: Check warnings with backslashes in keys
    marks = {
        '\.': r'. ',
        ',': r', ',
        '\?': r'? ',
        '!': r'! ',
        ':': r': ',
        ';': r'; '
    }
    for key, value in marks.items():
        # print(key)
        comment = re.sub(key, value, comment)
    return comment


def clean_register(comment, set_capital_letter, set_i_letter):
    """
    This function set lower case to all text

    Args:
    - comment - raw comment string
    - set_capital_letter - change the first sensen letter to upper case
    - set_i_letter - change i-letter to upper case

    Returns modified comment
    """
    comment = clean_punctuation_spaces(comment)
    comment = clean_spaces(comment)
    comment = comment.lower()
    sentences = re.split(r'[\.?!] ', comment)
    comment = ''
    if set_capital_letter:
        for sent in sentences:
            comment += sent.capitalize() + ' '
    comment = re.sub(r'[\.?!,:;\n"-]', '', comment)
    if set_i_letter:
        comment = re.sub(r' i ', ' I ', comment)
        comment = re.sub(r' i\'', ' I\'', comment)
    return comment


def clean_emoji(comment):
    """
    This function delete all emojis from comment

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    # TODO: Must be checked. If any errors occurs, remove r prefix
    pattern = re.compile(
        u"(?<!&)#(\w|(?:[\xA9\xAE\u203C\u2049\u2122\u2139\u2194-\u2199\u21A9\u21AA\u231A\u231B\u2328\u2388\u23CF\u23E9-\u23F3\u23F8-\u23FA\u24C2\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE\u2600-\u2604\u260E\u2611\u2614\u2615\u2618\u261D\u2620\u2622\u2623\u2626\u262A\u262E\u262F\u2638-\u263A\u2648-\u2653\u2660\u2663\u2665\u2666\u2668\u267B\u267F\u2692-\u2694\u2696\u2697\u2699\u269B\u269C\u26A0\u26A1\u26AA\u26AB\u26B0\u26B1\u26BD\u26BE\u26C4\u26C5\u26C8\u26CE\u26CF\u26D1\u26D3\u26D4\u26E9\u26EA\u26F0-\u26F5\u26F7-\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]|\uD83C[\uDC04\uDCCF\uDD70\uDD71\uDD7E\uDD7F\uDD8E\uDD91-\uDD9A\uDE01\uDE02\uDE1A\uDE2F\uDE32-\uDE3A\uDE50\uDE51\uDF00-\uDF21\uDF24-\uDF93\uDF96\uDF97\uDF99-\uDF9B\uDF9E-\uDFF0\uDFF3-\uDFF5\uDFF7-\uDFFF]|\uD83D[\uDC00-\uDCFD\uDCFF-\uDD3D\uDD49-\uDD4E\uDD50-\uDD67\uDD6F\uDD70\uDD73-\uDD79\uDD87\uDD8A-\uDD8D\uDD90\uDD95\uDD96\uDDA5\uDDA8\uDDB1\uDDB2\uDDBC\uDDC2-\uDDC4\uDDD1-\uDDD3\uDDDC-\uDDDE\uDDE1\uDDE3\uDDEF\uDDF3\uDDFA-\uDE4F\uDE80-\uDEC5\uDECB-\uDED0\uDEE0-\uDEE5\uDEE9\uDEEB\uDEEC\uDEF0\uDEF3]|\uD83E[\uDD10-\uDD18\uDD80-\uDD84\uDDC0]|(?:0\u20E3|1\u20E3|2\u20E3|3\u20E3|4\u20E3|5\u20E3|6\u20E3|7\u20E3|8\u20E3|9\u20E3|#\u20E3|\\*\u20E3|\uD83C(?:\uDDE6\uD83C(?:\uDDEB|\uDDFD|\uDDF1|\uDDF8|\uDDE9|\uDDF4|\uDDEE|\uDDF6|\uDDEC|\uDDF7|\uDDF2|\uDDFC|\uDDE8|\uDDFA|\uDDF9|\uDDFF|\uDDEA)|\uDDE7\uD83C(?:\uDDF8|\uDDED|\uDDE9|\uDDE7|\uDDFE|\uDDEA|\uDDFF|\uDDEF|\uDDF2|\uDDF9|\uDDF4|\uDDE6|\uDDFC|\uDDFB|\uDDF7|\uDDF3|\uDDEC|\uDDEB|\uDDEE|\uDDF6|\uDDF1)|\uDDE8\uD83C(?:\uDDF2|\uDDE6|\uDDFB|\uDDEB|\uDDF1|\uDDF3|\uDDFD|\uDDF5|\uDDE8|\uDDF4|\uDDEC|\uDDE9|\uDDF0|\uDDF7|\uDDEE|\uDDFA|\uDDFC|\uDDFE|\uDDFF|\uDDED)|\uDDE9\uD83C(?:\uDDFF|\uDDF0|\uDDEC|\uDDEF|\uDDF2|\uDDF4|\uDDEA)|\uDDEA\uD83C(?:\uDDE6|\uDDE8|\uDDEC|\uDDF7|\uDDEA|\uDDF9|\uDDFA|\uDDF8|\uDDED)|\uDDEB\uD83C(?:\uDDF0|\uDDF4|\uDDEF|\uDDEE|\uDDF7|\uDDF2)|\uDDEC\uD83C(?:\uDDF6|\uDDEB|\uDDE6|\uDDF2|\uDDEA|\uDDED|\uDDEE|\uDDF7|\uDDF1|\uDDE9|\uDDF5|\uDDFA|\uDDF9|\uDDEC|\uDDF3|\uDDFC|\uDDFE|\uDDF8|\uDDE7)|\uDDED\uD83C(?:\uDDF7|\uDDF9|\uDDF2|\uDDF3|\uDDF0|\uDDFA)|\uDDEE\uD83C(?:\uDDF4|\uDDE8|\uDDF8|\uDDF3|\uDDE9|\uDDF7|\uDDF6|\uDDEA|\uDDF2|\uDDF1|\uDDF9)|\uDDEF\uD83C(?:\uDDF2|\uDDF5|\uDDEA|\uDDF4)|\uDDF0\uD83C(?:\uDDED|\uDDFE|\uDDF2|\uDDFF|\uDDEA|\uDDEE|\uDDFC|\uDDEC|\uDDF5|\uDDF7|\uDDF3)|\uDDF1\uD83C(?:\uDDE6|\uDDFB|\uDDE7|\uDDF8|\uDDF7|\uDDFE|\uDDEE|\uDDF9|\uDDFA|\uDDF0|\uDDE8)|\uDDF2\uD83C(?:\uDDF4|\uDDF0|\uDDEC|\uDDFC|\uDDFE|\uDDFB|\uDDF1|\uDDF9|\uDDED|\uDDF6|\uDDF7|\uDDFA|\uDDFD|\uDDE9|\uDDE8|\uDDF3|\uDDEA|\uDDF8|\uDDE6|\uDDFF|\uDDF2|\uDDF5|\uDDEB)|\uDDF3\uD83C(?:\uDDE6|\uDDF7|\uDDF5|\uDDF1|\uDDE8|\uDDFF|\uDDEE|\uDDEA|\uDDEC|\uDDFA|\uDDEB|\uDDF4)|\uDDF4\uD83C\uDDF2|\uDDF5\uD83C(?:\uDDEB|\uDDF0|\uDDFC|\uDDF8|\uDDE6|\uDDEC|\uDDFE|\uDDEA|\uDDED|\uDDF3|\uDDF1|\uDDF9|\uDDF7|\uDDF2)|\uDDF6\uD83C\uDDE6|\uDDF7\uD83C(?:\uDDEA|\uDDF4|\uDDFA|\uDDFC|\uDDF8)|\uDDF8\uD83C(?:\uDDFB|\uDDF2|\uDDF9|\uDDE6|\uDDF3|\uDDE8|\uDDF1|\uDDEC|\uDDFD|\uDDF0|\uDDEE|\uDDE7|\uDDF4|\uDDF8|\uDDED|\uDDE9|\uDDF7|\uDDEF|\uDDFF|\uDDEA|\uDDFE)|\uDDF9\uD83C(?:\uDDE9|\uDDEB|\uDDFC|\uDDEF|\uDDFF|\uDDED|\uDDF1|\uDDEC|\uDDF0|\uDDF4|\uDDF9|\uDDE6|\uDDF3|\uDDF7|\uDDF2|\uDDE8|\uDDFB)|\uDDFA\uD83C(?:\uDDEC|\uDDE6|\uDDF8|\uDDFE|\uDDF2|\uDDFF)|\uDDFB\uD83C(?:\uDDEC|\uDDE8|\uDDEE|\uDDFA|\uDDE6|\uDDEA|\uDDF3)|\uDDFC\uD83C(?:\uDDF8|\uDDEB)|\uDDFD\uD83C\uDDF0|\uDDFE\uD83C(?:\uDDF9|\uDDEA)|\uDDFF\uD83C(?:\uDDE6|\uDDF2|\uDDFC))))[\ufe00-\ufe0f\u200d]?)+"
    )
    comment = pattern.sub('', comment)
    return comment


def clean_sites(comment):
    """
    This function delete all stop-words from comment

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    comment = re.sub(r'https|http|www|\\|/|_', ' ', comment)
    return comment


def clean_punctuation(comment):
    """
    It cleans punctuation

    Args:
    - comment - raw comment string

    Returns string without punctuation
    """
    comment = clean_sites(comment)
    comment = clean_emoji(comment)
    comment = clean_register(comment, True, False)
    comment = clean_spaces(comment)
    return comment


def clean_numbers(comment):
    """
    This function delete all numbers from comment

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    comment = re.sub(r'[ \n][\d+]+', '', comment)
    return comment


def clean_stop_words(comment):
    """
    This function delete all stop-words from comment

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    stop = set(stopwords.words('english'))
    new_comment = ' '.join(
        [i for i in comment.decode('utf-8').lower().split() if i not in stop])
    return new_comment


def clean_separate_letter(comment):
    """
    This function delete separate letter
    Example: "w o r l d"

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    comment = re.sub(r'[ \n][a-zA-Z]{1}[ \n]', '  ', comment)
    return comment


def correct_spelling(comment):
    """
    This function correct spelling

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    new_comment = ' '.join([spell(i) for i in comment.lower().split()])
    return new_comment


def parse_string_from_temp(comment):
    temp = re.split(r'\|', comment)
    if len(temp) > 1:
        del temp[0]

    new_string = ""
    for string in temp:
        temp_str = re.split(r'\=', string)
        if len(temp_str) > 1:
            del temp_str[0]
        for s in temp_str:
            new_string += s
            new_string += ' '
        if new_string[len(new_string) - 1] == ' ':
            new_string = new_string[:-1]
    return new_string


def delete_wiki_templates(comment):
    """
    This function delete wiki_templates

    Args:
    - comment - raw comment string

    Returns modified comment
    """
    bracket_deque = deque()
    bracket_poses = deque()
    i = 0
    new_comment = ""
    while i < len(comment):
        ch = comment[i]
        if ch == '{' or ch == '[':
            bracket_deque.append(ch)
            bracket_poses.append(i)
        elif ch == '}':
            if bracket_deque and bracket_deque.pop() == '{':
                pos = bracket_poses.pop()
                new_comment = comment[:pos]
                new_comment += parse_string_from_temp(comment[pos + 1:i])
                new_comment += comment[i + 1:]
                i -= len(comment) - len(new_comment)
                comment = new_comment
        elif ch == ']':
            if bracket_deque and bracket_deque.pop() == '[':
                pos = bracket_poses.pop()
                new_comment = comment[:pos]
                new_comment += parse_string_from_temp(comment[pos + 1:i])
                new_comment += comment[i + 1:]
                i -= len(comment) - len(new_comment) - 2
                comment = new_comment

        i += 1
    comment = re.sub(r'[{}\[\]]', '', comment)
    return comment


def clean_comment(comment):
    """
    It cleans comment

    Args:
    - comment - raw comment string.raw

    Returns clean comment string in utf-8. Original `comment` is not transformed
    """
    new_clean_comment = clean_separate_letter(comment)
    # TODO: strange line. Must be checked!
    # new_clean_comment = delete_wiki_templates(comment)
    new_clean_comment = delete_wiki_templates(new_clean_comment)
    new_clean_comment = clean_punctuation(new_clean_comment)
    new_clean_comment = clean_numbers(new_clean_comment)
    new_clean_comment = clean_stop_words(new_clean_comment)
    # new_clean_comment = correct_spelling(new_clean_comment)
    return new_clean_comment


def process(series):
    # pbar.update(1)
    return u'\"{}\"'.format(clean_comment(series['comment_text']))


def call_process(df):
    return df.apply(process, axis=1)


def clean(data, cpus, stage=None):
    """It cleans comments from test.csv"""
    # pbar = tqdm(total=data.shape[0], desc=stage)
    pool = Pool(processes=cpus)
    data_split = np.array_split(data, cpus)
    pool_results = pool.map(call_process, data_split)
    # pbar.close()
    data['comment_text'] = pd.concat(pool_results)
    return data


def main():
    """Main function"""

    # pull argparams
    parser = init_argparse()
    args = parser.parse_args()

    try_makedirs(args.processed)

    clean(
        pd.read_csv(args.train),
        args.cpus,
        stage='Cleaning {}'.format(args.train)).to_csv(
            path.join(args.processed, path.basename(args.train)),
            index=False,
            encoding='utf-8')

    clean(
        pd.read_csv(args.test),
        args.cpus,
        stage='Cleaning {}'.format(args.test)).to_csv(
            path.join(args.processed, path.basename(args.test)),
            index=False,
            encoding='utf-8')


if __name__ == '__main__':
    main()
