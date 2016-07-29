#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import codecs
from collections import namedtuple
from subprocess import Popen, PIPE

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
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

SENNA_PATH = os.path.expanduser('~/src/thesis/senna/')

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
                    dataset.data.append({'text': text})
                except ValueError:
                    try:
                        [uid, polarity, text] = line.split(separator,
                                                           maxsplit=3)
                        if ignore_not_available and text == 'Not Available':
                            continue
                        dataset.uid.append(uid)
                        dataset.target_names.append(polarity.replace('"', ''))
                        dataset.data.append({'text': text})
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
class ItemExtractor(BaseEstimator, TransformerMixin):
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


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """Extract main features.

    Attributes:
        features: A dictionnary with features to extract of the form
        {feature_name: extractor} where extractor is a function.
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        ret = []
        for x in X:
            d = {}
            for (f_name, f) in self.features:
                extracted = f(x)
                if type(extracted) is dict:
                    d.update(extracted)
                else:
                    d[f_name] = extracted
            ret.append(d)
        return ret


##### Caps
def f_all_caps(s):
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


def f_senna(s):
    """Return the string parsed with senna.

    Args:
        s: A string to parse.

    Returns:
        A dictionnary with POS, CHK and NER annotations for each word.

    Raises:
        IOError: An error occurred.
    """
    ret = {'POS': [], 'CHK': [], 'NER': []}
    p = Popen(['./senna', '-notokentags',
               '-usrtokens',
               '-pos', '-chk', '-ner'],
              stdin=PIPE,
              stdout=PIPE,
              stderr=PIPE, cwd=SENNA_PATH)
    out, err = p.communicate(input=s.encode())
    if p.returncode != 0:
        print(err.replace('*', '#'))
    out = out.decode()

    for line in out.split('\n'):
        line = line.strip()
        if not line:
            continue
        tags = line.split('\t')
        tags = [x.strip() for x in tags]
        pos, chk, ner = tags
        ret['POS'].append(pos)
        ret['CHK'].append(chk)
        ret['NER'].append(ner)
    return ret


def f_senna_multilines(s):
    """Return the string parsed with senna.

    Args:
        s: A string to parse.

    Returns:
        A list of dictionnaries with POS, CHK and NER annotations for
        each word.

    Raises:
        IOError: An error occurred.
    """
    ret = []
    p = Popen(['./senna', '-notokentags',
               '-usrtokens',
               '-pos', '-chk', '-ner'],
              stdin=PIPE,
              stdout=PIPE,
              stderr=PIPE, cwd=SENNA_PATH)
    out, err = p.communicate(input=s.encode())
    if p.returncode != 0:
        print(err.replace('*', '#'))
    out = out.decode()

    d = {'POS': [], 'CHK': [], 'NER': []}
    new_line = True
    for line in out.split('\n'):
        line = line.strip()
        if not line:
            if not new_line:
                ret.append(d)
                d = {'POS': [], 'CHK': [], 'NER': []}
            new_line = True
            continue
        tags = line.split('\t')
        tags = [x.strip() for x in tags]
        pos, chk, ner = tags
        d['POS'].append(pos)
        d['CHK'].append(chk)
        d['NER'].append(ner)
        new_line = False
    return ret


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


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


class Tokenizer(BaseEstimator, TransformerMixin):
    """Tokenize with the given tokenizer."""
    def __init__(self, item=None, tokenizer=None):
        self.item = item
        self.tokenizer = tokenizer
        self._update_params()

    def _update_params(self):
        if self.tokenizer is None:
            self.tokenizer = identity

    def fit(self, X, y=None, *params):
        return self

    def transform(self, X, *params):
        if self.item is None:
            return [self.tokenizer(x) for x in X]
        else:
            for x in X:
                x[self.item] = self.tokenizer(x[self.item])
        return X

    def get_params(self, deep=True):
        return {'item': self.item,
                'tokenizer': self.tokenizer}

    def set_params(self, **params):
        for p in params:
            setattr(self, p, params[p])
        self._update_params()


class Filter(BaseEstimator, TransformerMixin):
    """Filter input based."""
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None, *params):
        return self

    def transform(self, X, *params):
        if self.enabled:
            return X
        else:
            return [[0] for x in X]

    def get_params(self, deep=True):
        return {'enabled': self.enabled}

    def set_params(self, **params):
        for p in params:
            setattr(self, p, params[p])


########## Pipeline
def run(truncate=None):
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

                  'features__ngram__vect__ngram_range': [(1, 3)],
                  'features__ngram__tfidf__use_idf': [True],

                  'clf__alpha': [1e-4],
                  'clf__n_iter': [5],
                  'clf__loss': ['hinge'],
                  }

    pipeline = Pipeline([
        ('tok', Tokenizer('text')),
        ('features', FeatureUnion([
            ('ngram', Pipeline([
                ('extract', ItemExtractor('text')),
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
            ])),
        ])),
        ('clf', SGDClassifier(loss='hinge', random_state=42)),
    ])

    scorer = metrics.make_scorer(metrics.f1_score,
                                 average='micro')
    scorer = metrics.make_scorer(metrics.accuracy_score)
    scorer = 'accuracy'
    clf = GridSearchCV(pipeline, parameters, n_jobs=1,
                       scoring=scorer, verbose=1)
    clf = clf.fit(train.data[:truncate], train.target[:truncate])
    for param in clf.best_params_:
        print('%s: %r' % (param, clf.best_params_[param]))

    # clf = pipeline
    # clf = clf.fit(train.data, train.target)

    predicted = clf.predict(test.data)
    print(metrics.classification_report(test.target, predicted,
                                        target_names=labels))
