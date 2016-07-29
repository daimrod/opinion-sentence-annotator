#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import codecs
from collections import namedtuple

from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import happyfuntokenizing

from nltk.tokenize import TweetTokenizer


logger = logging.getLogger(__name__)

## Configuration
import os

if 'DATA_DIR' in os.environ:
    DATA_DIR = os.environ['DATA_DIR']
else:
    DATA_DIR = '/media/jadi-g/DATADRIVE1/corpus'

semeval13 = os.path.join(DATA_DIR, 'SEMEVAL13')
semeval13_polarity_train = os.path.join(semeval13, 'tweeti-b.dist.tsv.2')
semeval13_polarity_dev = os.path.join(semeval13, 'tweeti-b.dev.dist.tsv.2')
semeval16 = os.path.join(DATA_DIR, 'SEMEVAL16')
semeval16_polarity_train = os.path.join(semeval16,
                                        'Task4',
                                        '100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt')
semeval16_polarity_test = os.path.join(semeval16,
                                       'Task4',
                                       '100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt')

########## Data Reader
Dataset = namedtuple('Dataset', ['data',
                                 'filenames',  # we don't need this for semeval
                                 'target_names',
                                 'target',
                                 'uid',  'sid',  # we need those for semeval
                                 ])


def read_dataset(ipath, separator='\t',
                 ignore_not_available=True):
    """Return a dataset following sklearn format.

       Convert the data from SEMEVAL Sentiment Analysis Task of
       Message Polarity Detection to sklearn format.

       The sklearn format is presented here :
       http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

       train.data[N] = the data themselves, e.g. the strings
       train.filenames[N] = the filename from wich the data come from
       train.target_names[N] = the name of the class associated with each entry
       train.target[N] = the class of each entry converted to integer

       The format of the SEMEVAL dataset for the Message Polarity task
       can be the following :
       <SID><tab><UID><tab><TOPIC><tab><positive|negative|neutral|objective><tab><TWITTER_MESSAGE>
       <UID><tab><TOPIC><tab><positive|negative|neutral|objective><tab><TWITTER_MESSAGE>
       Where the polarity can be between double-quotes or not and TEXT
       is the content of the tweet or 'Not Available' if the tweet
       doesn't exist anymore.

    Args:
        ipath: The path to the dataset in SEMEVAL format.
        separator: The separator between attributes in a line.
        ignore_not_available: Set to True to ignore tweets that aren't
        available in the SEMEVAL corpus (we weren't able to download
        them).
    Returns:
        A Dataset with filled with the data from ipath.

    Raises:
        IOError: An error occurred.
    """
    dataset = Dataset(data=[],
                      filenames=[],
                      target_names=[],
                      target=[],
                      sid=[],
                      uid=[])
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        for line in ifile:
            line = line.strip()
            if line:
                try:
                    [sid, uid, polarity, text] = line.split(separator,
                                                            maxsplit=4)
                    if ignore_not_available and text == 'Not Available':
                        continue
                    dataset.sid.append(sid)
                    dataset.uid.append(uid)
                    dataset.target_names.append(polarity.replace('"', ''))
                    dataset.data.append(text)
                except ValueError:
                    try:
                        [uid, polarity, text] = line.split(separator,
                                                           maxsplit=3)
                        if ignore_not_available and text == 'Not Available':
                            continue
                        dataset.uid.append(uid)
                        dataset.target_names.append(polarity.replace('"', ''))
                        dataset.data.append(text)
                    except ValueError:
                        logging.warn('Couldn\'t parse line %s', line)
    return dataset


def strings_to_integers(strings):
    """Convert an array of strings to an array of integers.

    Convert an array of strings to an array of integers where the same
    string will always have the same integers value.

    Args:
        strings: An array of strings.

    Returns:
        An array of integers
    """
    labels = list(set(strings))
    integers = []
    for string in strings:
        integers.append(labels.index(string))
    return integers, labels


def merge_classes(lst, classes, new_class):
    """Merge classes from lst into one new_class.

    Args:
        lst: A list of classes that will be replaced (strings).
        classes: A list of classes to replace (strings).
        new_class: The new class (string).

    Returns:
        The list with all occurences of classes replaced by new_class.
    """
    for i in range(len(lst)):
        if lst[i] in classes:
            lst[i] = new_class
    return lst


########## Features Extractions
##### Utils
class ItemExtractor(TransformerMixin):
    """Extract a particular entry in the input dictionnary.

    Attributes:
        item: The item to extract.
    """
    def __init__(self, item):
        self.item = item

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        return [x[self.item] for x in X]


##### Caps
def all_caps(s):
    """Return the number of words with all characters in upper case.

    This function assumes that the string is tokenized and all words
    are separated by spaces.

    Args:
        s: A string.

    Returns:
        The number of words with all characters in upper case.
    """
    n = 0
    for word in s.split(' '):
        if word.upper() == word:
            n = n + 1
    return n


##### Tokenizer
def happyfuntokenizer(s):
    """Tokenize a string with happyfuntokenizing.py.

    Args:
        s: A string to tokenize.

    Returns:
        The string tokenized.
    """
    tok = happyfuntokenizing.Tokenizer()
    return ' '.join(tok.tokenize(s))


def nltktokenizer(s):
    """Tokenize a string with NLTK Twitter Tokenizer.

    Args:
        s: A string to tokenize.

    Returns:
        The string tokenized.
    """
    tok = TweetTokenizer()
    return ' '.join(tok.tokenize(s))


def ident(*args):
    return args


class Tokenizer(BaseEstimator, TransformerMixin):
    """Tokenize.

    """
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tok = ident
        else:
            self.tok = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, arg):
        return arg


########## Pipeline
def run():
    train = read_dataset(semeval16_polarity_train)
    # Convert objective and neutral to objective/neutral
    merge_classes(train.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    # Build the target array
    target, labels = strings_to_integers(train.target_names)
    train.target.extend(target)

    test = read_dataset(semeval16_polarity_test)
    # Convert objective and neutral to objective/neutral
    merge_classes(test.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    # Build the target array
    target, labels = strings_to_integers(test.target_names)
    test.target.extend(target)

    # parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #               'tfidf__use_idf': [True, False],
    #               'clf__alpha': [1e-3, 1e-4, 1e-5],
    #               'clf__n_iter': [1, 2, 5],
    #               'clf__loss': ['hinge'],
    # }

    parameters = {'tok__tokenizer': [None, happyfuntokenizer, nltktokenizer],

                  'vect__ngram_range': [(1, 3)],

                  'tfidf__use_idf': [True],

                  'clf__alpha': [1e-4],
                  'clf__n_iter': [5],
                  'clf__loss': ['hinge'],
                  }

    pipeline = Pipeline([('tok', Tokenizer()),
                         ('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', random_state=42)),
                         ])

    scorer = metrics.make_scorer(metrics.f1_score,
                                 average='micro')
    scorer = metrics.make_scorer(metrics.accuracy_score)
    scorer = 'accuracy'
    clf = GridSearchCV(pipeline, parameters, n_jobs=6,
                       scoring=scorer, verbose=1)
    clf = clf.fit(train.data, train.target)
    for param in clf.best_params_:
        print('%s: %r' % (param, clf.best_params_[param]))

    # clf = pipeline
    # clf = clf.fit(train.data, train.target)

    predicted = clf.predict(test.data)
    print(metrics.classification_report(test.target, predicted,
                                        target_names=labels))
