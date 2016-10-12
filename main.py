#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import codecs
from subprocess import Popen, PIPE

import tempfile

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import re

import os

import happyfuntokenizing

from nltk.tokenize import TweetTokenizer

if 'logger' not in locals():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # FileHandler
    fh = logging.FileHandler('log.txt', 'a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

## Configuration
if 'DATA_DIR' in os.environ:
    DATA_DIR = os.environ['DATA_DIR']
else:
    DATA_DIR = '/media/jadi-g/DATADRIVE1/corpus'

### Stanford Tools
if 'CLASSPATH' not in os.environ:
    os.environ['CLASSPATH'] = ''
os.environ['CLASSPATH'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-parser-full-2015-12-09/') + ':' + os.environ['CLASSPATH']
os.environ['CLASSPATH'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-postagger-full-2015-12-09/') + ':' + os.environ['CLASSPATH']

if 'STANFORD_MODELS' not in os.environ:
    os.environ['STANFORD_MODELS'] = ''
os.environ['STANFORD_MODELS'] = os.path.expanduser('~/src/java/stanford-for-nltk/') + ':' + os.environ['STANFORD_MODELS']
os.environ['STANFORD_MODELS'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-parser-full-2015-12-09/') + ':' + os.environ['STANFORD_MODELS']
os.environ['STANFORD_MODELS'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-postagger-full-2015-12-09/models/') + ':' +os.environ['STANFORD_MODELS']


### Dataset
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

old_rouvier_train = os.path.join(os.path.expanduser('~/src/thesis/Rouvier-SemEval2016/data/'), 'train.txt')
old_rouvier_test = os.path.join(os.path.expanduser('~/src/thesis/Rouvier-SemEval2016/data/'), 'twitter16.txt')
rouvier_train = os.path.join(semeval16, 'Task4', 'rouvier_train.txt')
rouvier_test = os.path.join(semeval16, 'Task4', 'rouvier_twitter16.txt')

# train_path = semeval16_polarity_train
# test_path = semeval16_polarity_devtest
train_path = rouvier_train
test_path = rouvier_test

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
def f_neg_context(s):
    """Return an enriched representation of the input string with negated
contexts.

    Add _NEG to words in negated contexts (see
    http://sentiment.christopherpotts.net/lingstruc.html).

    Args:
        s: A string.

    Returns:
        An enriched representation of the input string with negation contexts.

    """
    re_beg_ctxt = r"(\b(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b|n't\b)"
    re_end_ctxt = r"[.:;!?]+"
    s_with_neg = []
    in_neg_ctxt = False
    for word in s.split(' '):
        if re.search(re_end_ctxt, word, flags=re.IGNORECASE):
            in_neg_ctxt = False
        elif re.search(re_beg_ctxt, word, flags=re.IGNORECASE):
            in_neg_ctxt = True
        elif in_neg_ctxt:
            word = word + '_NEG'
        s_with_neg.append(word)
    return ' '.join(s_with_neg)


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
    return [n]


def f_n_hashtags(s):
    """Return the number of hashtags in the string.

    Args:
        s: A string.

    Returns:
        The number of hashtags.
    """
    n = 0
    for word in s.split(' '):
        if word.startswith('#'):
            n += 1
    return [n]


def f_punctuation(s):
    """Return information about the punctuation in the input string.

    - the number of contiguous sequences of exclamation marks,
      question marks, and both exclamation and question marks;
    - whether the last token contains exclamation or question mark;

    Args:
        s: variable documentation.

    Returns:
        - the number of contiguous sequences of exclamation marks,
          question marks, and both exclamation and question marks;
        - whether the last token contains exclamation or question mark;
    """
    continuous_excl = 0
    continuous_quest = 0
    continuous_excl_quest = 0

    for word in s.split(' '):
        excl_flag = 0
        quest_flag = 0
        excl_quest_flag = 0
        for char in word:
            if char == '!':
                excl_flag += 1
            else:
                excl_flag = 0
            if char == '?':
                quest_flag += 1
            else:
                quest_flag = 0
            if char == '!' or char == '?':
                excl_quest_flag += 1
            else:
                excl_quest_flag = 0
        else:
            if excl_flag > 1:
                continuous_excl += 1
            if quest_flag > 1:
                continuous_quest += 1
            if excl_quest_flag > 1:
                continuous_excl_quest += 1
    last_word = s.split(' ')[-1]
    last_excl_or_quest = '!' in last_word or '?' in last_word

    return [continuous_excl,
            continuous_quest,
            continuous_excl_quest,
            last_excl_or_quest * 1]


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
    return [n]


def f_all_syntax(s):
    ret = []
    for f in [f_elgongated_words,
              f_all_caps]:
        ret.extend(f(s))
    return ret


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


def pretty_pipeline(obj):
    if isinstance(obj, list):
        return [pretty_pipeline(o) for o in obj]
    elif isinstance(obj, FeatureUnion):
        return {'n_jobs': obj.n_jobs,
                'transformer_list': obj.transformer_list,
                'transformer_weights': obj.transformer_weights}
    elif isinstance(obj, Pipeline):
        return {'steps': pretty_pipeline(obj.steps)}
    elif isinstance(obj, tuple):
        return pretty_pipeline(list(obj))
    else:
        return obj


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
def run(truncate=None, test_dataset=None):
    logger.info('Read train dataset')
    logger.debug(train_path)
    train = read_dataset(train_path)
    logger.info('  Convert objective and neutral to objective/neutral')
    merge_classes(train.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    logger.info('  Build the target array')
    target, labels = strings_to_integers(train.target_names)
    train.target.extend(target)
    train.data = train.data[:truncate]
    train.target = train.target[:truncate]

    logger.info('Read test dataset')
    logger.debug(test_path)
    test = read_dataset(test_path)
    if test_dataset is not None:
        logger.info('  Filter test dataset (keep only %s)' % test_dataset)
        indexes = [idx for (idx, val) in enumerate(test.uid)
                   if val == test_dataset]
        new_uid = []
        new_sid = []
        new_data = []
        new_target_names = []
        for idx in indexes:
            new_uid.append(test.uid[idx])
            new_sid.append(test.sid[idx])
            new_data.append(test.data[idx])
            new_target_names.append(test.target_names[idx])
        test.uid = new_uid
        test.sid = new_sid
        test.data = new_data
        test.target_names = new_target_names

    logger.info('  Convert objective and neutral to objective/neutral')
    merge_classes(test.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    logger.info('  Build the target array')
    target, labels = strings_to_integers(test.target_names)
    test.target.extend(target)

    logger.info('Tokenize text')
    for d in train.data:
        d['tok'] = happyfuntokenizer(d['text'])
    for d in test.data:
        d['tok'] = happyfuntokenizer(d['text'])

    logger.info('Extract Senna features')
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

    logger.info('Identify negated contexts')
    for d in train.data:
        d['tok'] = f_neg_context(d['tok'])
    for d in test.data:
        d['tok'] = f_neg_context(d['tok'])

    logger.info('Train the pipeline')
    clf = Pipeline([
        ('text_features', FeatureUnion(
            [('vect', Pipeline([
                ('selector', ItemExtractor('tok')),
                ('vect', CountVectorizer())])),
             ('tfidf', Pipeline([
                 ('selector', ItemExtractor('tok')),
                 ('tfidf', TfidfVectorizer())])),
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
             ('twitter', Pipeline([
                 ('selector', ItemExtractor('tok')),
                 ('hashtags', ApplyFunction(f_n_hashtags))])),
             ('punctuation', Pipeline([
                 ('selector', ItemExtractor('tok')),
                 ('punctuation', ApplyFunction(f_punctuation))])),
            ])),
        ('clf', SGDClassifier(loss='hinge',
                              n_iter=5,
                              random_state=42))]).fit(train.data, train.target)

    logger.info('Classify test data')
    predicted = clf.predict(test.data)
    logger.info('Results')
    logger.debug(pretty_pipeline(clf))
    logger.info('\n' +
                metrics.classification_report(test.target, predicted,
                                              target_names=list(set(test.target_names))))
    return clf
