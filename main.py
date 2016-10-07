#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import codecs
from collections import namedtuple
from subprocess import Popen, PIPE

import tempfile

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
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
semeval16_polarity_devtest = os.path.join(semeval16,
                                       'Task4',
                                       '100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt')

rouvier_train_ = os.path.join(os.path.expanduser('~/src/thesis/Rouvier-SemEval2016/data/'), 'train.txt')
rouvier_test = os.path.join(os.path.expanduser('~/src/thesis/Rouvier-SemEval2016/data/'), 'twitter16.txt')

train_path = semeval16_polarity_train
test_path = semeval16_polarity_devtest

SENNA_PATH = os.path.expanduser('~/src/thesis/senna/')


########## Data Reader
class Dataset():
    def __init__(self,
                 data=[],
                 filenames=[],
                 target_names=[],
                 target=[],
                 uid=[], sid=[]):
        self.data = data
        self.filenames = filenames
        self.target_names = target_names
        self.target = target
        self.uid = uid
        self.sid = sid


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


class ApplyFunction(BaseEstimator, TransformerMixin):
    """Apply a function to each entry.

    Attributes:
        fun: The function to apply.
    """
    def __init__(self, fun):
        self.fun = fun

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        return [self.fun(x) for x in X]


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


def f_elgongated_words(s):
    """Return the number of words with one character repeated more than 2
times.

    This function assumes that the string is tokenized and all words
    are separated by spaces.

    Args:
        s: A string.

    Returns:
        The number of words with one character repeated more than 2 times.
    """
    n = 0
    for word in s.split(' '):
        for i in range(len(word) - 2):
            s = word[i:3+i]
            if len(set(s)) == 1:
                n = n + 1
                break
    return n


def f_all_syntax(s):
    return [f(s) for f in [
        f_elgongated_words,
        f_all_caps,
    ]]


##### Senna
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


def f_senna_multi(lst):
    """Return the list of strings parsed with senna.

    Args:
        lst: A list of strings to parse.

    Returns:
        A list of dictionnaries with POS, CHK and NER annotations for
        each word.

    Raises:
        IOError: An error occurred.
    """
    ret = []
    with tempfile.TemporaryFile() as temp_file:
        for s in lst:
            temp_file.write(s.encode())
            temp_file.write(b'\n')
        temp_file.seek(0, 0)
        p = Popen(['./senna', '-notokentags',
                   '-usrtokens',
                   '-pos', '-chk', '-ner'],
                  stdin=temp_file,
                  stdout=PIPE,
                  stderr=PIPE, cwd=SENNA_PATH)
    out, err = p.communicate(input=s.encode())
    if p.returncode != 0:
        print(err.replace('*', '#'))
    out = out.decode()

    d = {'pos': [], 'chk': [], 'ner': []}
    for line in out.split('\n'):
        line = line.strip()
        if not line:
            if d['pos'] != []:
                d['pos'] = ' '.join(d['pos'])
                d['chk'] = ' '.join(d['chk'])
                d['ner'] = ' '.join(d['ner'])
                ret.append(d)
                d = {'pos': [], 'chk': [], 'ner': []}
            continue
        tags = line.split('\t')
        tags = [x.strip() for x in tags]
        pos, chk, ner = tags
        d['pos'].append(pos)
        d['chk'].append(chk)
        d['ner'].append(ner)
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


def dummy(*args):
    return 'dummy'


def string_to_feature(s, prefix):
    f = {}
    for (idx, val) in enumerate(s.split()):
        f['%s-%d' % (prefix, idx)] = val
    return f


def apply_pipeline(pipeline, train, test):
    fitted = pipeline.fit(train)
    return (fitted.transform(train), fitted.transform(test))


########## Pipeline
def run(truncate=None):
    train = read_dataset(train_path)
    # Convert objective and neutral to objective/neutral
    merge_classes(train.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    # Build the target array
    target, labels = strings_to_integers(train.target_names)
    train.target.extend(target)
    train.data = train.data[:truncate]
    train.target = train.target[:truncate]

    test = read_dataset(test_path)
    # Convert objective and neutral to objective/neutral
    merge_classes(test.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')

    # Build the target array
    target, labels = strings_to_integers(test.target_names)
    test.target.extend(target)

    # Tokenize text
    for d in train.data:
        d['tok'] = happyfuntokenizer(d['text'])
    for d in test.data:
        d['tok'] = happyfuntokenizer(d['text'])

    # Extract Senna
    senna = f_senna_multi([d['tok'] for d in train.data])
    for idx in range(len(train.data)):
        train.data[idx]['pos'] = senna[idx]['pos']
        train.data[idx]['chk'] = senna[idx]['chk']
        train.data[idx]['ner'] = senna[idx]['ner']

    senna = f_senna_multi([d['tok'] for d in test.data])
    for idx in range(len(test.data)):
        test.data[idx]['pos'] = senna[idx]['pos']
        test.data[idx]['chk'] = senna[idx]['chk']
        test.data[idx]['ner'] = senna[idx]['ner']

    # parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #               'tfidf__use_idf': [True, False],
    #               'clf__alpha': [1e-3, 1e-4, 1e-5],
    #               'clf__n_iter': [1, 2, 5],
    #               'clf__loss': ['hinge'],
    # }

    # parameters = {'tok__fun': [identity, happyfuntokenizer, nltktokenizer],

    #               'features__ngram__vect__ngram_range': [(1, 3)],
    #               'features__ngram__tfidf__use_idf': [True],

    #               'clf__alpha': [1e-4],
    #               'clf__n_iter': [5],
    #               'clf__loss': ['hinge'],
    #               }

    clf = Pipeline([
        ('text_features', FeatureUnion(
            [('vect', Pipeline([
                ('selector', ItemExtractor('tok')),
                ('vect', CountVectorizer())])),
             ('tfidf', Pipeline([
                 ('selector', ItemExtractor('tok')),
                 ('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer())])),
             ('pos', Pipeline([
                 ('selector', ItemExtractor('pos')),
                 ('vect', CountVectorizer(binary=True))])),
             ('chk', Pipeline([
                 ('selector', ItemExtractor('chk')),
                 ('vect', CountVectorizer(binary=True))])),
             ('ner', Pipeline([
                 ('selector', ItemExtractor('ner')),
                 ('vect', CountVectorizer(binary=True))])),
             ('syntax', Pipeline([
                 ('selector', ItemExtractor('tok')),
                 ('syntax', ApplyFunction(f_all_syntax))])),
            ])),
        ('clf', SGDClassifier(loss='hinge',
                              n_iter=5,
                              random_state=42))]).fit(train.data, train.target)
    predicted = clf.predict(test.data)
    print(metrics.classification_report(test.target, predicted,
                                        target_names=list(set(test.target_names))))


    # scorer = metrics.make_scorer(metrics.f1_score,
    #                              average='micro')
    # scorer = metrics.make_scorer(metrics.accuracy_score)
    # scorer = 'accuracy'
    # clf = GridSearchCV(pipeline, parameters, n_jobs=1,
    #                    scoring=scorer, verbose=1)
    # clf = clf.fit(train.data, train.target)
    # for param in clf.best_params_:
    #     print('%s: %r' % (param, clf.best_params_[param]))

    # clf = pipeline
    # clf = clf.fit(train.data, train.target)
