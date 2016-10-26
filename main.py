#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pickle
import os

import gensim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD

from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import read_bing_liu
from reader import read_carnegie_clusters
from reader import read_mpqa
from reader import read_semeval_dataset
from reader import read_nrc_hashtag
from reader import read_nrc_emotion
from reader import TwitterLoggerTextReader

from reader import Tokenizer
from reader import Splitter
from reader import LexiconProjecter
from reader import URLReplacer
from reader import UserNameReplacer
from utils import merge_classes
from utils import pretty_pipeline
from utils import strings_to_integers
from utils import eval_with_semeval_script

import features as feat
import resources as res

if 'logger' not in locals():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() %(levelname)-8s %(message)s')
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
    dataset.labels = labels
    dataset.target.extend(target)

    logger.info('Normalize and tokenize the text')
    generator = (d['text'] for d in dataset.data)
    preprocessor = URLReplacer(generator)
    preprocessor = UserNameReplacer(preprocessor)
    preprocessor = Tokenizer(preprocessor, feat.happyfuntokenizer)
    for (d, tok) in zip(dataset.data, preprocessor):
        d['tok'] = tok

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


def runNRCCanada(train_truncate=None, test_truncate=None,
                 test_dataset=None, new_text_features=[]):
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
    logger.info('Load the corpus')
    with open(preprocess(res.train_path, force=False), 'rb') as p_file:
        train = pickle.load(p_file)

    with open(preprocess(res.test_path, force=False), 'rb') as p_file:
        test = pickle.load(p_file)
    train.truncate(train_truncate)
    test.truncate(test_truncate)

    logger.info('Load the resources')
    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                     res.bing_liu_lexicon_path['positive'])
    nrc_emotion_lexicon = read_nrc_emotion(res.nrc_emotion_lexicon_path)
    nrc_hashtag_lexicon = read_nrc_hashtag(res.nrc_hashtag_lexicon_path)
    mpqa_lexicon = read_mpqa(res.mpqa_lexicon_path)
    carnegie_clusters = read_carnegie_clusters(res.carnegie_clusters_path)

    logger.info('Build the pipeline')
    text_features = [
        ('all caps', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('all caps', feat.ApplyFunction(feat.f_all_caps))])),
        ('clusters', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('clusters', feat.ApplyFunction(feat.IM_Project_Lexicon(carnegie_clusters))),
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
            ('projection', feat.ApplyFunction(feat.F_NRC_Project_Lexicon(nrc_emotion_lexicon)))])),
        ('nrc_hashtag_lexicon', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('projection', feat.ApplyFunction(feat.F_NRC_Project_Lexicon(nrc_hashtag_lexicon)))])),
        ('bing_liu_lexicon', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('projection', feat.ApplyFunction(feat.F_NRC_Project_Lexicon(bing_liu_lexicon)))])),
        ('mpqa_lexicon', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('projection', feat.ApplyFunction(feat.F_NRC_Project_Lexicon(mpqa_lexicon)))])),

        ('word ngram', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('tfidf', CountVectorizer(binary=True,
                                      ngram_range=(1, 4)))])),
        ('char ngram', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('tfidf', CountVectorizer(binary=True, analyzer='char',
                                      ngram_range=(3, 5)))]))]
    text_features.extend(new_text_features)

    logger.info('Train the pipeline')
    clf = Pipeline([
        ('text_features', FeatureUnion(text_features)),
        ('clf', SGDClassifier(loss='hinge',
                              n_iter=5,
                              n_jobs=5,
                              random_state=42))]).fit(train.data, train.target)

    logger.info('Classify test data')
    predicted = clf.predict(test.data)
    logger.info('Results')
    logger.debug(pretty_pipeline(clf))
    logger.info('\n' +
                metrics.classification_report(test.target, predicted,
                                              target_names=list(set(test.target_names))))

    logger.info('\n' +
                eval_with_semeval_script(predicted, test))
    return clf, predicted


def runCustom0(train_truncate=None, test_truncate=None, test_dataset=None):
    """
    """
    logger.info('Load the resources')
    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                     res.bing_liu_lexicon_path['positive'])
    word2vec_path = res.twitter_logger_en_path + '.word2vec'
    if os.path.exists(word2vec_path) and os.path.getmtime(word2vec_path) > os.path.getmtime(res.twitter_logger_en_path):
        word2vec = gensim.models.Word2Vec.load(word2vec_path)
    else:
        reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
        reader = URLReplacer(reader)
        reader = UserNameReplacer(reader)
        reader = Tokenizer(reader, feat.happyfuntokenizer)
        reader = Splitter(reader)
        word2vec = gensim.models.Word2Vec(reader, min_count=10, workers=4)
        word2vec.init_sims(replace=True)
        word2vec.save(word2vec_path)

    text_features = [
        ('word2vec find closest', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('find closest', feat.ApplyFunction(feat.F_Find_Closest_In_Lexicon(word2vec, bing_liu_lexicon))),
            ('convert to feature', feat.ApplyFunction(feat.Array_To_Feature('word2vec-closest'))),
            ('use feature', DictVectorizer())]))]

    return runNRCCanada(train_truncate=train_truncate,
                        test_truncate=test_truncate,
                        test_dataset=test_dataset,
                        new_text_features=text_features)


def runCustom0_with_SVD(train_truncate=None, test_truncate=None, test_dataset=None):
    """
    """
    logger.info('Load the resources')
    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                     res.bing_liu_lexicon_path['positive'])
    word2vec_path = res.twitter_logger_en_path + '.word2vec'
    if os.path.exists(word2vec_path) and os.path.getmtime(word2vec_path) > os.path.getmtime(res.twitter_logger_en_path):
        word2vec = gensim.models.Word2Vec.load(word2vec_path)
    else:
        reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
        reader = URLReplacer(reader)
        reader = UserNameReplacer(reader)
        reader = Tokenizer(reader, feat.happyfuntokenizer)
        reader = Splitter(reader)
        word2vec = gensim.models.Word2Vec(reader, min_count=10, workers=4)
        word2vec.init_sims(replace=True)
        word2vec.save(word2vec_path)

    text_features = [
        ('word2vec find closest', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('find closest', feat.ApplyFunction(feat.F_Find_Closest_In_Lexicon(word2vec, bing_liu_lexicon))),
            ('convert to feature', feat.ApplyFunction(feat.Array_To_Feature('word2vec-closest'))),
            ('use feature', DictVectorizer()),
            ('SVD', TruncatedSVD(n_components=100))]))]

    return runNRCCanada(train_truncate=train_truncate,
                        test_truncate=test_truncate,
                        test_dataset=test_dataset,
                        new_text_features=text_features)


def runCustom1(train_truncate=None, test_truncate=None, test_dataset=None):
    """
    """
    logger.info('Load the resources')
    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                     res.bing_liu_lexicon_path['positive'])
    word2vec_path = res.twitter_logger_en_path + '.word2vec.custom1'
    if os.path.exists(word2vec_path) and os.path.getmtime(word2vec_path) > os.path.getmtime(res.twitter_logger_en_path):
        word2vec = gensim.models.Word2Vec.load(word2vec_path)
    else:
        logger.info('Train word2vec model')
        reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
        reader = URLReplacer(reader)
        reader = UserNameReplacer(reader)
        reader = Tokenizer(reader, feat.happyfuntokenizer)
        reader = Splitter(reader)
        reader = LexiconProjecter(reader, bing_liu_lexicon)
        word2vec = gensim.models.Word2Vec(reader, min_count=10, workers=4)
        word2vec.init_sims(replace=True)
        word2vec.save(word2vec_path)

    text_features = [
        ('word2vec find closest', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('find closest', feat.ApplyFunction(feat.F_Find_Closest_In_Lexicon(word2vec, bing_liu_lexicon))),
            ('convert to feature', feat.ApplyFunction(feat.Array_To_Feature('word2vec-closest'))),
            ('use feature', DictVectorizer())]))]

    return runNRCCanada(train_truncate=train_truncate,
                        test_truncate=test_truncate,
                        test_dataset=test_dataset,
                        new_text_features=text_features)


def runCustom1_with_SVD(train_truncate=None,
                        test_truncate=None,
                        test_dataset=None):
    """
    """
    logger.info('Load the resources')
    bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                     res.bing_liu_lexicon_path['positive'])
    word2vec_path = res.twitter_logger_en_path + '.word2vec.custom1'
    if os.path.exists(word2vec_path) and os.path.getmtime(word2vec_path) > os.path.getmtime(res.twitter_logger_en_path):
        word2vec = gensim.models.Word2Vec.load(word2vec_path)
    else:
        logger.info('Train word2vec model')
        reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
        reader = URLReplacer(reader)
        reader = UserNameReplacer(reader)
        reader = Tokenizer(reader, feat.happyfuntokenizer)
        reader = Splitter(reader)
        reader = LexiconProjecter(reader, bing_liu_lexicon)
        word2vec = gensim.models.Word2Vec(reader, min_count=10, workers=4)
        word2vec.init_sims(replace=True)
        word2vec.save(word2vec_path)

    text_features = [
        ('word2vec find closest', Pipeline([
            ('selector', feat.ItemExtractor('tok')),
            ('find closest', feat.ApplyFunction(feat.F_Find_Closest_In_Lexicon(word2vec, bing_liu_lexicon))),
            ('convert to feature', feat.ApplyFunction(feat.Array_To_Feature('word2vec-closest'))),
            ('use feature', DictVectorizer()),
            ('SVD', TruncatedSVD(n_components=100))]))]

    return runNRCCanada(train_truncate=train_truncate,
                        test_truncate=test_truncate,
                        test_dataset=test_dataset,
                        new_text_features=text_features)
