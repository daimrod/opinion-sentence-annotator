#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import codecs

import pickle
import os
import ast

import gensim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import read_bing_liu
from reader import read_carnegie_clusters
from reader import read_mpqa
from reader import read_semeval_dataset
from reader import read_nrc_hashtag
from reader import read_nrc_emotion

from utils import merge_classes
from utils import pretty_pipeline
from utils import strings_to_integers

import features as feat
import resources as res

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


########## Pipeline
class TwitterLoggerTextReader(object):
    """Read tweets as recorded by my twitter-logger project.

    Tweets are saved using the following format :
    ('id', u'ID')\t('text', u'TWEET')

    Attributes:
        filename: The path to the tweets.
    """
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            line = line.strip()
            elements = line.split('\t')
            if len(elements) != 2:
                logger.error('Incorrect formatting %s', line)
                continue
            _, t_text = elements
            _, text = ast.literal_eval(t_text)
            yield text


class Tokenizer(object):
    def __init__(self, iterable, tokenizer):
        self.iterable = iterable
        self.tokenizer = tokenizer

    def __iter__(self):
        for s in self.iterable:
            yield self.tokenizer(s)


class Splitter(object):
    def __init__(self, iterable, split=' '):
        self.iterable = iterable
        self.split = split

    def __iter__(self):
        for s in self.iterable:
            yield s.split(self.split)


class LexiconProjecter(object):
    """
    """
    def __init__(self, iterable, lexicon):
        self.iterable = iterable
        self.lexicon = lexicon

    def __iter__(self):
        for s in self.iterable:
            new_s = []
            for w in s:
                if w in self.lexicon:
                    new_s.append(self.lexicon[w])
                else:
                    new_s.append(w)
            yield new_s


def preprocess(dataset_path, force=False):
    preprocessed_path = dataset_path + '.pp.pickle'
    if not force and os.path.isfile(preprocessed_path):
        return preprocessed_path

    logger.info('Read dataset')
    logger.debug(dataset_path)
    dataset = read_semeval_dataset(dataset_path)
    logger.info('  Convert objective and neutral to objective/neutral')
    merge_classes(dataset.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    logger.info('  Build the target array')
    target, labels = strings_to_integers(dataset.target_names)
    dataset.target.extend(target)

    logger.info('Tokenize text')
    for d in dataset.data:
        d['tok'] = feat.happyfuntokenizer(d['text'])

    logger.info('Extract Senna features')
    senna = feat.f_senna_multi([d['tok'] for d in dataset.data])
    for idx in range(len(dataset.data)):
        dataset.data[idx]['pos'] = senna[idx]['pos']
        dataset.data[idx]['chk'] = senna[idx]['chk']
        dataset.data[idx]['ner'] = senna[idx]['ner']

    logger.info('Identify negated contexts')
    for d in dataset.data:
        d['neg_tok'] = feat.im_neg_context(d['tok'])

    with open(preprocessed_path, 'wb') as p_file:
        pickle.dump(dataset, p_file)
    return preprocessed_path


def train():
    pass


def test():
    pass


def runNRCCanada(truncate=None, test_dataset=None):
    """Reimplementation of NRCCanada

http://www.saifmohammad.com/WebPages/Abstracts/NRC-SentimentAnalysis.htm

FEATURES:

For tweet-level sentiment detection:
- all-caps: the number of words with all characters in upper case;
- clusters: presence/absence of tokens from each of the 1000 clusters
  (provided by Carnegie Mellon University's Twitter NLP tool);
- elongated words: the number of words with one character repeated
  more than 2 times, e.g. 'soooo';
- emoticons:
      - presence/absence of positive and negative emoticons at any
        position in the tweet;
      - whether the last token is a positive or negative emoticon;
- hashtags: the number of hashtags;
- negation: the number of negated contexts. A negated context also
  affects the ngram and lexicon features: each word and associated
  with it polarity in a negated context become negated (e.g., 'not
  perfect' becomes 'not perfect_NEG', 'POLARITY_positive' becomes
  'POLARITY_positive_NEG');
- POS: the number of occurrences for each part-of-speech tag;
- punctuation:
      - the number of contiguous sequences of exclamation marks,
        question marks, and both exclamation and question marks;
      - whether the last token contains exclamation or question mark;
- sentiment lexicons: automatically created lexicons (NRC Hashtag
  Sentiment Lexicon, Sentiment140 Lexicon), manually created sentiment
  lexicons (NRC Emotion Lexicon, MPQA, Bing Liu Lexicon). For each
  lexicon and each polarity we calculated:
      - total count of tokens in the tweet with score greater than 0;
      - the sum of the scores for all tokens in the tweet;
      - the maximal score;
      - the non-zero score of the last token in the tweet;
      The lexicon features were created for all tokens in the tweet,
      for each part-of-speech tag, for hashtags, and for all-caps
      tokens.
- word ngrams, character ngrams.
    """
    with open(preprocess(res.train_path, force=False), 'rb') as p_file:
        train = pickle.load(p_file)

    with open(preprocess(res.test_path, force=False), 'rb') as p_file:
        test = pickle.load(p_file)

    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'], res.bing_liu_lexicon_path['positive'])
    nrc_emotion_lexicon = read_nrc_emotion(res.nrc_emotion_lexicon_path)
    nrc_hashtag_lexicon = read_nrc_hashtag(res.nrc_hashtag_lexicon_path)
    mpqa_lexicon = read_mpqa(res.mpqa_lexicon_path)
    carnegie_clusters = read_carnegie_clusters(res.carnegie_clusters_path)

    logger.info('Train the pipeline')
    clf = Pipeline([
        ('text_features', FeatureUnion(
            [('all caps', Pipeline([
                ('selector', feat.ItemExtractor('tok')),
                ('all caps', feat.ApplyFunction(feat.f_all_caps))])),
             ('clusters', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('clusters', feat.ApplyFunction(feat.make_im_project_lexicon(carnegie_clusters))),
                 ('convertion', CountVectorizer(binary=True))])),
             ('elongated', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('elongated', feat.ApplyFunction(feat.f_elgongated_words))])),
             ('emoticons', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('emoticons', feat.ApplyFunction(feat.f_emoticons))])),
             ('hashtags', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('hashtags', feat.ApplyFunction(feat.f_n_hashtags))])),
             ('negation', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('negation', feat.ApplyFunction(feat.f_n_neg_context))])),
             ('pos', Pipeline([
                 ('selector', feat.ItemExtractor('pos')),
                 ('vect', CountVectorizer())])),
             ('punctuation', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('punctuation', feat.ApplyFunction(feat.f_punctuation))])),
             ('nrc_emotion_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(nrc_emotion_lexicon)))])),
             ('nrc_hashtag_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(nrc_hashtag_lexicon)))])),
             ('bing_liu_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(bing_liu_lexicon)))])),
             ('mpqa_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(mpqa_lexicon)))])),


             ('word ngram', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('tfidf', CountVectorizer(binary=True,
                                           ngram_range=(1, 4)))])),
             ('char ngram', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('tfidf', CountVectorizer(binary=True, analyzer='char',
                                           ngram_range=(3, 5)))])),
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


def runCustom0(train_truncate=None, test_truncate=None, test_dataset=None):
    """
    """
    logger.info('Load the training file')
    with open(preprocess(res.train_path, force=False), 'rb') as p_file:
        train = pickle.load(p_file)

    logger.info('Load the testing file')
    with open(preprocess(res.test_path, force=False), 'rb') as p_file:
        test = pickle.load(p_file)
    train.truncate(train_truncate)
    test.truncate(test_truncate)

    logger.info('Load the ressources')
    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'], res.bing_liu_lexicon_path['positive'])
    nrc_emotion_lexicon = read_nrc_emotion(res.nrc_emotion_lexicon_path)
    nrc_hashtag_lexicon = read_nrc_hashtag(res.nrc_hashtag_lexicon_path)
    mpqa_lexicon = read_mpqa(res.mpqa_lexicon_path)
    carnegie_clusters = read_carnegie_clusters(res.carnegie_clusters_path)
    if os.path.exists(res.twitter_logger_en_path + '.word2vec'):
        word2vec = gensim.models.Word2Vec.load(res.twitter_logger_en_path + '.word2vec')
    else:
        reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
        reader = Tokenizer(reader, feat.happyfuntokenizer)
        reader = Splitter(reader)
        word2vec = gensim.models.Word2Vec(reader, min_count=10, workers=4)
        word2vec.init_sims(replace=True)
        word2vec.save(res.twitter_logger_en_path + '.word2vec')

    logger.info('Train the pipeline')
    clf = Pipeline([
        ('text_features', FeatureUnion(
            [('all caps', Pipeline([
                ('selector', feat.ItemExtractor('tok')),
                ('all caps', feat.ApplyFunction(feat.f_all_caps))])),
             ('clusters', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('clusters', feat.ApplyFunction(feat.make_im_project_lexicon(carnegie_clusters))),
                 ('convertion', CountVectorizer(binary=True))])),
             ('elongated', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('elongated', feat.ApplyFunction(feat.f_elgongated_words))])),
             ('emoticons', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('emoticons', feat.ApplyFunction(feat.f_emoticons))])),
             ('hashtags', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('hashtags', feat.ApplyFunction(feat.f_n_hashtags))])),
             ('negation', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('negation', feat.ApplyFunction(feat.f_n_neg_context))])),
             ('pos', Pipeline([
                 ('selector', feat.ItemExtractor('pos')),
                 ('vect', CountVectorizer())])),
             ('punctuation', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('punctuation', feat.ApplyFunction(feat.f_punctuation))])),
             ('nrc_emotion_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(nrc_emotion_lexicon)))])),
             ('nrc_hashtag_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(nrc_hashtag_lexicon)))])),
             ('bing_liu_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(bing_liu_lexicon)))])),
             ('mpqa_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(mpqa_lexicon)))])),
             ('word2vec find closest', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('find closest', feat.ApplyFunction(feat.make_f_find_closest_in_lexicon(word2vec, bing_liu_lexicon))),
                 ('convert to feature', feat.ApplyFunction(feat.make_array_to_feature('word2vec-closest'))),
                 ('use feature', DictVectorizer())])),

             ('tfidf', Pipeline([
                 ('selector', feat.ItemExtractor('neg_tok')),
                 ('tfidf', TfidfVectorizer())])),
             # ('chk', Pipeline([
             #     ('selector', feat.ItemExtractor('chk')),
             #     ('vect', CountVectorizer(binary=True))])),
             # ('ner', Pipeline([
             #     ('selector', feat.ItemExtractor('ner')),
             #     ('vect', CountVectorizer(binary=True))])),
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


def runCustom1(train_truncate=None, test_truncate=None, test_dataset=None):
    """
    """
    logger.info('Load the training file')
    with open(preprocess(res.train_path, force=False), 'rb') as p_file:
        train = pickle.load(p_file)

    logger.info('Load the testing file')
    with open(preprocess(res.test_path, force=False), 'rb') as p_file:
        test = pickle.load(p_file)
    train.truncate(train_truncate)
    test.truncate(test_truncate)

    logger.info('Load the ressources')
    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'], res.bing_liu_lexicon_path['positive'])
    nrc_emotion_lexicon = read_nrc_emotion(res.nrc_emotion_lexicon_path)
    nrc_hashtag_lexicon = read_nrc_hashtag(res.nrc_hashtag_lexicon_path)
    mpqa_lexicon = read_mpqa(res.mpqa_lexicon_path)
    carnegie_clusters = read_carnegie_clusters(res.carnegie_clusters_path)
    word2vec_path = res.twitter_logger_en_path + '.word2vec.custom1'
    if os.path.exists(word2vec_path):
        word2vec = gensim.models.Word2Vec.load(word2vec_path)
    else:
        logger.info('Train word2vec model')
        reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
        reader = Tokenizer(reader, feat.happyfuntokenizer)
        reader = Splitter(reader)
        reader = LexiconProjecter(reader, bing_liu_lexicon)
        word2vec = gensim.models.Word2Vec(reader, min_count=10, workers=4)
        word2vec.init_sims(replace=True)
        word2vec.save(word2vec_path)

    logger.info('Train the pipeline')
    clf = Pipeline([
        ('text_features', FeatureUnion(
            [('all caps', Pipeline([
                ('selector', feat.ItemExtractor('tok')),
                ('all caps', feat.ApplyFunction(feat.f_all_caps))])),
             ('clusters', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('clusters', feat.ApplyFunction(feat.make_im_project_lexicon(carnegie_clusters))),
                 ('convertion', CountVectorizer(binary=True))])),
             ('elongated', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('elongated', feat.ApplyFunction(feat.f_elgongated_words))])),
             ('emoticons', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('emoticons', feat.ApplyFunction(feat.f_emoticons))])),
             ('hashtags', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('hashtags', feat.ApplyFunction(feat.f_n_hashtags))])),
             ('negation', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('negation', feat.ApplyFunction(feat.f_n_neg_context))])),
             ('pos', Pipeline([
                 ('selector', feat.ItemExtractor('pos')),
                 ('vect', CountVectorizer())])),
             ('punctuation', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('punctuation', feat.ApplyFunction(feat.f_punctuation))])),
             ('nrc_emotion_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(nrc_emotion_lexicon)))])),
             ('nrc_hashtag_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(nrc_hashtag_lexicon)))])),
             ('bing_liu_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(bing_liu_lexicon)))])),
             ('mpqa_lexicon', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('projection', feat.ApplyFunction(feat.make_f_nrc_project_lexicon(mpqa_lexicon)))])),
             ('word2vec find closest', Pipeline([
                 ('selector', feat.ItemExtractor('tok')),
                 ('find closest', feat.ApplyFunction(feat.make_f_find_closest_in_lexicon(word2vec, bing_liu_lexicon))),
                 ('convert to feature', feat.ApplyFunction(feat.make_array_to_feature('word2vec-closest'))),
                 ('use feature', DictVectorizer())])),

             ('tfidf', Pipeline([
                 ('selector', feat.ItemExtractor('neg_tok')),
                 ('tfidf', TfidfVectorizer())])),
             # ('chk', Pipeline([
             #     ('selector', feat.ItemExtractor('chk')),
             #     ('vect', CountVectorizer(binary=True))])),
             # ('ner', Pipeline([
             #     ('selector', feat.ItemExtractor('ner')),
             #     ('vect', CountVectorizer(binary=True))])),
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
