# coding=utf-8
"""
Cleans comments

Usage: python text_cleaner.py [-h]
Also, it can be imported into another module
"""
import argparse
import HTMLParser
from string import punctuation
from itertools import groupby
from os import path
from multiprocessing import Process, Pool, cpu_count
import pandas as pd
import re
import numpy as np
from utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, AUGMENTED_TRAIN_FILES, \
                  try_makedirs, get_stop_words

HTML_PARSER = HTMLParser.HTMLParser()
STOP_WORDS = get_stop_words()
"""
the web url matching regex used by markdown
http://daringfireball.net/2010/07/improved_regex_for_matching_urls
https://gist.github.com/gruber/8891611
"""
URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
EMOJI_PATTERN = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+",
    flags=re.UNICODE)
MARKS = {r'[\s]+.': '. ', r',': ', ', r'?': '? ', r'!': '! '}
PUNCTUATION = set(punctuation) # string of ASCII punctuation


def init_argparse():
    """
    Initializes argparse

    Returns parser
    """
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
        '--load_augmented',
        help='Use augmente data for training',
        action='store_true')
    parser.add_argument(
        '--cpus',
        nargs='?',
        help='Number of CPUs to use. By default, it will use all CPUs',
        default=cpu_count(),
        type=int)
    return parser


def remove_code_sequencies(comment, html=True, wiki_templates=False):
    """
    Removes html code and wiki templates

    Returns clean comment
    """
    clean_comment = comment
    if html:
        clean_comment = HTML_PARSER.unescape(comment)
    if wiki_templates:
        # TODO Clean wiki templates
        pass
    return clean_comment


def remove_punctuation(comment):
    """
    Removes all punctuations except !?,.'.

    Returns clean comment
    """
    # removes other punctuation
    new_comment = re.sub(r"[^\w\s!?,.']", ' ', comment)
    # removes duplicate punctuation
    new_comment = re.sub(r"([\s!?,.'])\1+", r'\1', new_comment)
    # remove spaces before punctuation
    new_comment = re.sub(r"[\s]+(?=[!.,'])", '', new_comment)
    # add space after punctuation
    new_comment = re.sub(r"([!.,'])(?=[\w\d])", r'\1 ', new_comment)
    clean_comment = new_comment.strip()
    return clean_comment


def remove_emojis(comment):
    """
    Removes emojis from the comment

    Returns clean comment
    """
    return EMOJI_PATTERN.sub(' ', comment)


def split_attached_words(comment):
    """
    HelloToEveryone -> Hello To Everyone

    Returns clean comment
    """
    return ' '.join(re.findall('[A-Z][^A-Z]*', comment))


def remove_urls(comment):
    """
    Removes URLs

    Returns clean comment
    """
    return re.sub(URL_REGEX, '', comment)


def standardize_words(comment):
    """
    happpy -> happy :-), looooove -> loove :-(

    Returns clean comment
    """
    return ''.join(''.join(s)[:2] for _, s in groupby(comment))


def remove_digits(comment):
    """
    Removes digits

    Returns clean comment
    """
    new_comment = []
    for word in comment.split():
        # Filter out punctuation and stop words
        if (not word.lstrip('-').replace('.', '', 1).isdigit()):
            new_comment.append(word)
    clean_comment = ' '.join(new_comment)
    return clean_comment


def remove_stop_words(comment):
    """
    Removes all stop-words and separate numbers from comment

    Returns clean comment
    """
    new_comment = []
    for word in comment.split():
        # It filters out digits, punctuation and stop words
        if (not word.replace('.', '', 1).isdigit() and
                word not in PUNCTUATION and word.lower() not in STOP_WORDS):
            new_comment.append(word)
    return ' '.join(new_comment)


def clean_comment(comment):
    """
    Cleans comment

    Returns clean comment string in utf-8.
    Original raw `comment` is not transformed
    """
    clean_comment = remove_code_sequencies(
        comment, html=True, wiki_templates=False)
    clean_comment = remove_urls(comment)
    clean_comment = remove_emojis(clean_comment)
    clean_comment = standardize_words(clean_comment)
    clean_comment = remove_punctuation(clean_comment)
    clean_comment = remove_digits(clean_comment)
    clean_comment = remove_stop_words(clean_comment)
    # remove punctuatiion once more
    clean_comment = remove_punctuation(clean_comment)
    return clean_comment


def process(series):
    return u'\"{}\"'.format(clean_comment(series['comment_text']))


def call_process(df):
    return df.apply(process, axis=1)


def clean(raw_file, processed_file, cpus):
    """
    Cleans comments from test.csv
    """
    data = pd.read_csv(raw_file, encoding='utf-8')
    pool = Pool(processes=cpus)
    data_split = np.array_split(data, cpus)
    pool_results = pool.map(call_process, data_split)
    data['comment_text'] = pd.concat(pool_results)
    data.to_csv(processed_file, index=False, encoding='utf-8')


def main():
    """
    Main function
    """

    # pull argparams
    parser = init_argparse()
    args = parser.parse_args()

    try_makedirs(args.processed)

    test_process = Process(
        target=clean,
        args=(
            args.test,
            path.join(args.processed, path.basename(args.test)),
            args.cpus,))
    train_processes = [
        Process(
            target=clean,
            args=(
                args.train,
                path.join(args.processed, path.basename(args.train)),
                args.cpus,))
    ]
    if args.load_augmented:
        raw_prefix = path.splitext(args.train)[0]
        processed_prefix = path.join(args.processed, path.basename(raw_prefix))
        for suffix in AUGMENTED_TRAIN_FILES:
            train_processes.append(
                Process(
                    target=clean,
                    args=(
                        raw_prefix + suffix,
                        processed_prefix + suffix,
                        args.cpus,)))

    test_process.start()
    for proc in train_processes:
        proc.start()
    test_process.join()
    for proc in train_processes:
        proc.join()


if __name__ == '__main__':
    main()
