"""
Prints some information about training data

Usage: python info.py
"""
import codecs
import string
import re
from threading import Thread
from os import path
import nltk
import pandas as pd
import numpy as np
from utils import RAW_DATA_PATH, LOG_PATH, try_makedirs


def freq_dist(train_data, result, top=50, low=50):
    """Returns the most and the least popular words in `data`"""
    data = train_data['comment_text'].str.encode('utf-8').to_string()

    # NLTK's default English stopwords
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    custom_stopwords = set(
        codecs.open(path.join(RAW_DATA_PATH, 'stopwords.txt'), 'r', 'utf-8')
        .read().splitlines())
    stopwords = default_stopwords | custom_stopwords
    words = nltk.word_tokenize(data)

    # Remove single-character tokens (mostly punctuation)
    # Possible issue because some 'w o r d s' will be removed
    words = [word for word in words if len(word) > 1]

    # Remove numbers, '\\n', hex numbers starts with \\x
    words = [re.sub(r'\d+|\\n|\\x[0-9a-f]+', '', word) for word in words]

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [word.translate(table) for word in words]
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

    result['freq_dist'] = [fdist.most_common(top), fdist.most_common()[-low:]]


def freq_labels(train_data, result):
    """Returns data frame that contains label and its frequency"""
    freq = {}
    for _, row in train_data[[
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
            'identity_hate'
    ]].iterrows():
        label = ''.join(str(c) for c in row)
        if label in freq:
            freq[label] += 1
        else:
            freq[label] = 1
    series = pd.Series(freq, name='freq')
    series.index.name = 'label'
    result['freq_labels'] = series.reset_index().sort_values(
        by=['freq', 'label'], ascending=False)


def create_freq_dist(data, file_name):
    """Creates word_freq file"""
    with open(path.join(LOG_PATH, file_name), 'w') as file:
        file.write('word,freq\n')
        for word, freq in data:
            file.write(str(word))
            file.write(',')
            file.write(str(freq))
            file.write('\n')


def count_comment_statistics(data, file_name):
    """
    It finds the longest and shortes comments
    and counts some other statistics about comment length
    """

    def plot_distribution():
        """It saves comment length distribution plot into png file"""
        import matplotlib
        # generates images without having a window appear
        matplotlib.use('Agg')
        import matplotlib.pylab as plt

        plt.hist(comments, bins=20)
        plt.title("Comment length distribution")
        plt.savefig(
            path.join(LOG_PATH, '{}.png'.format(path.splitext(file_name)[0])))

    comments = np.vectorize(len)(data.as_matrix(columns=['comment_text']))
    with open(path.join(LOG_PATH, file_name), 'w') as file:
        file.write('metrics,value\n')
        file.write('min,{}\n'.format(comments.min()))
        file.write('mean,{0:.2f}\n'.format(comments.mean()))
        file.write('std,{0:.2f}\n'.format(comments.std()))
        file.write('max,{}\n'.format(comments.max()))
    plot_distribution()


def main():
    """Main function"""
    train_data = pd.read_csv(path.join(RAW_DATA_PATH, 'train.csv'))
    result = {}

    count_comments = Thread(
        target=count_comment_statistics, args=(train_data, 'comments.csv'))
    count_comments.start()

    get_labels = Thread(target=freq_labels, args=(train_data, result))
    get_labels.start()

    get_dict = Thread(target=freq_dist, args=(train_data, result, 100, 100))
    get_dict.start()

    count_comments.join()
    get_labels.join()
    get_dict.join()
    print(result['freq_labels'].to_string(index=False))
    try_makedirs(LOG_PATH)
    result['freq_labels'].to_csv(
        path.join(LOG_PATH, 'freq_labels.csv'), index=False)
    most, least = result['freq_dist']
    print('\nMost:\n', most)
    print('\nLeast:\n', least)
    create_freq_dist(most, 'most_freq_dist.csv')
    create_freq_dist(least, 'least_freq_dist.csv')


if __name__ == '__main__':
    main()
