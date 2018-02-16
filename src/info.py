"""
Prints some information about training data

Usage: python info.py
"""
from argparse import ArgumentParser
import codecs
import string
import re
from threading import Thread
from os import path
import nltk
import pandas as pd
from utils import RAW_DATA_PATH, LOG_PATH, RAW_DATA_PATH, \
                  try_makedirs, load_train_data


def freq_dist(train_data, result, top=50, low=50):
    """Returns the most and the least popular words in `data`"""
    data = train_data['comment_text'].str.decode('utf-8').to_string()

    # NLTK's default English stopwords
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    # pull auxiliary stopwords from user file
    custom_stopwords = set(
        codecs.open(path.join(RAW_DATA_PATH, 'stopwords.txt'), 'r', 'utf-8')
        .read().splitlines())
    # generate set of all using stopwords
    stopwords = default_stopwords | custom_stopwords

    words = nltk.word_tokenize(data)

    # Remove single-character tokens (mostly punctuation)
    # Possible issue because some 'w o r d s' will be removed
    words = [word for word in words if len(word) > 1]

    # Remove numbers, '\\n', hex numbers starts with \\x
    words = [re.sub(r'\d+|\\n|\\x[0-9a-f]+', '', word) for word in words]

    # Remove punctuation
    stripped = [word.translate(string.punctuation) for word in words]
    # Remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # Lowercase all words (default_stopwords are lowercase too)
    words = [word.lower() for word in words]

    # Stemming words
    # stemmer = nltk.stem.snowball.SnowballStemmer('english')
    # words = [stemmer.stem(word) for word in words]

    # Remove stopwords
    words = [word for word in words if word not in stopwords]

    # Calculate frequency distribution
    fdist = nltk.FreqDist(words)

    # the most and the least popular words
    result['freq_dist'] = [fdist.most_common(top), fdist.most_common()[-low:]]


def freq_labels(train_data, result):
    """Returns data frame that contains labels and their frequencies"""
    freq = {}
    for _, row in train_data[[
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
            'identity_hate'
    ]].iterrows():
        # generate united label name from set of labels
        label = ''.join(str(c) for c in row)
        # calculate values of label frequences
        if label in freq:
            freq[label] += 1
        else:
            freq[label] = 1
    series = pd.Series(freq, name='freq')
    series.index.name = 'label'
    # labels and their frequencies
    result['freq_labels'] = series.reset_index().sort_values(
        by=['freq', 'label'], ascending=False)


def create_freq_dist(data, file_name):
    """Creates word_freq file"""
    pddist = pd.DataFrame(data, columns=['word', 'freq'])
    pddist.to_csv(
        path.join(LOG_PATH, file_name), index=False, encoding='utf-8')


def count_comment_statistics(data, file_name):
    """
    It finds the longest and shortest comments
    and counts some other statistics about comment length
    """

    def plot_distribution():
        """It saves comment length distribution plot into png file"""
        import matplotlib
        # generates images without having a window appear
        matplotlib.use('Agg')
        import matplotlib.pylab as plt

        plt.hist(comments.as_matrix(), bins=20)
        plt.title('Comment length distribution')
        plt.xlabel('Number of comments')
        plt.ylabel('Number of words')
        plt.savefig(
            path.join(LOG_PATH, '{}.png'.format(path.splitext(file_name)[0])))

    comments = data['comment_text'].str.len()
    comments.describe().to_csv(
        path=path.join(LOG_PATH, file_name),
        index_label=['metrics', 'value'],
        float_format='%g',
        encoding='utf-8')
    plot_distribution()


def init_argparse():
    """Initializes argparse"""
    parser = ArgumentParser(description='Trains toxic comment classifier')
    parser.add_argument(
        '-t',
        '--train',
        nargs='?',
        help='path to train.csv file',
        default=path.join(RAW_DATA_PATH, 'train.csv'),
        type=str)
    parser.add_argument(
        '--load_augmented',
        help='Use augmente data for training',
        action='store_true')
    return parser


def main():
    """Main function"""
    args = init_argparse().parse_args()

    train_data = load_train_data(args.train, args.load_augmented)
    result = {}

    try_makedirs(LOG_PATH)
    count_comments = Thread(
        target=count_comment_statistics,
        args=(train_data, 'comments_statistics.csv'))
    count_comments.start()

    # pull class frequences of comments
    get_labels = Thread(target=freq_labels, args=(train_data, result))
    get_labels.start()

    # pull frequences of words from comments
    get_dict = Thread(target=freq_dist, args=(train_data, result, 100, 100))
    get_dict.start()

    count_comments.join()
    get_labels.join()
    get_dict.join()

    # output info about frequences of labels and words from comments
    print(result['freq_labels'].to_string(index=False))
    result['freq_labels'].to_csv(
        path.join(LOG_PATH, 'freq_labels.csv'), index=False, encoding='utf-8')
    most, least = result['freq_dist']
    print('\nMost frequent words:\n', most)
    print('\nLeast frequent words:\n', least)
    create_freq_dist(most, 'most_freq_dist.csv')
    create_freq_dist(least, 'least_freq_dist.csv')


if __name__ == '__main__':
    main()
