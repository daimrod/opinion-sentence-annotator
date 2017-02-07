#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pickle
import os

import gensim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD

from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import read_bing_liu
from reader import read_carnegie_clusters
from reader import read_mpqa
from reader import read_nrc_hashtag_unigram
from reader import read_nrc_hashtag_bigram
from reader import read_nrc_hashtag_pair
from reader import read_nrc_hashtag_sentimenthashtags
from reader import read_nrc_emotion
from reader import TwitterLoggerTextReader

from reader import Tokenizer
from reader import Splitter
from reader import LexiconProjecter
from reader import URLReplacer
from reader import UserNameReplacer

from utils import pretty_pipeline
from utils import eval_with_semeval_script
from utils import assoc_value

import features as feat
from features import ApplyFunction as AF
import resources as res
import embeddings as emb

from base import FullPipeline
from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from base import preprocess

SVMRegister = {}

logger = logging.getLogger(__name__)


class SmallPipeline(FullPipeline):
    def __init__(self,
                 train_truncate=None, test_truncate=None,
                 only_uid=None,
                 train_only_labels=['positive', 'negative', 'neutral'],
                 test_only_labels=['positive', 'negative', 'neutral'],
                 repreprocess=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_truncate = train_truncate
        self.test_truncate = test_truncate
        self.only_uid = only_uid
        self.train_only_labels = train_only_labels
        self.test_only_labels = test_only_labels
        self.repreprocess = repreprocess

    def load_resources(self):
        super().load_resources()
        logger.info('Load the corpus')
        with open(preprocess(res.train_path, force=self.repreprocess), 'rb') as p_file:
            self.train = pickle.load(p_file)
        with open(preprocess(res.test_path, force=self.repreprocess), 'rb') as p_file:
            self.test = pickle.load(p_file)
        self.train.truncate(self.train_truncate)
        self.test.truncate(self.test_truncate)
        self.train.filter_label(self.train_only_labels)
        self.test.filter_label(self.test_only_labels)
        if self.only_uid is not None:
            self.test.filter_uid(self.only_uid)

    def build_pipeline(self):
        super().build_pipeline()
        self.text_features = [
            ['all caps', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['all caps', AF(feat.f_all_caps)]])],
            ]

    def run_train(self):
        super().run_train()
        # clf = Pipeline([
        #     ('text_features', FeatureUnion(text_features)),
        #     #('standard scaler', StandardScaler(with_mean=False)),
        #     #('min/max scaler', MinMaxScaler()),
        #     ('max abs scaler', MaxAbsScaler()),
        #     ('clf', SVC(C=0.005, kernel='linear', max_iter=1000))]).fit(train.data, train.target)
        self.clf = Pipeline([
            ['text_features', FeatureUnion(self.text_features)],
            ['max abs scaler', MaxAbsScaler()],
            ['clf', SGDClassifier(loss='hinge',
                                  n_iter=100,
                                  n_jobs=5,
                                  # class_weight="balanced",
                                  # class_weight={0: 1, 1: 1, 2: 0.5},
                                  # class_weight={0: 1, 1: 1.5, 2: 0.25},
                                  random_state=42
            )]]).fit(self.train.data, self.train.target)

    def run_test(self):
        super().run_test()
        self.predicted = self.clf.predict(self.test.data)

    def print_results(self):
        super().print_results()
        logger.debug(pretty_pipeline(self.clf))
        logger.info('\n' +
                    metrics.classification_report(self.test.target, self.predicted,
                                                  target_names=self.test.labels))

        try:
            logger.info('\n' +
                        eval_with_semeval_script(self.test, self.predicted))
        except:
            pass

    def run(self):
        super().run()
        return self.clf, self.predicted, self.text_features


class NRCCanada(FullPipeline):
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
    def __init__(self,
                 train_truncate=None, test_truncate=None,
                 only_uid=None,
                 train_only_labels=['positive', 'negative', 'neutral'],
                 test_only_labels=['positive', 'negative', 'neutral'],
                 repreprocess=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_truncate = train_truncate
        self.test_truncate = test_truncate
        self.only_uid = only_uid
        self.train_only_labels = train_only_labels
        self.test_only_labels = test_only_labels
        self.repreprocess = repreprocess

    def load_resources(self):
        super().load_resources()
        logger.info('Load the corpus')
        with open(preprocess(res.train_path, force=self.repreprocess), 'rb') as p_file:
            self.train = pickle.load(p_file)
        with open(preprocess(res.test_path, force=self.repreprocess), 'rb') as p_file:
            self.test = pickle.load(p_file)
        self.train.truncate(self.train_truncate)
        self.test.truncate(self.test_truncate)
        self.train.filter_label(self.train_only_labels)
        self.test.filter_label(self.test_only_labels)
        if self.only_uid is not None:
            self.test.filter_uid(self.only_uid)

        logger.info('Load the lexicons')
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path)
        self.nrc_emotion_lexicon = read_nrc_emotion(res.nrc_emotion_lexicon_path)
        self.nrc_hashtag_unigram_lexicon = read_nrc_hashtag_unigram(res.nrc_hashtag_unigram_lexicon_path)
        self.nrc_hashtag_bigram_lexicon = read_nrc_hashtag_bigram(res.nrc_hashtag_bigram_lexicon_path)
        self.nrc_hashtag_pair_lexicon = read_nrc_hashtag_pair(res.nrc_hashtag_pair_lexicon_path)
        self.nrc_sentiment140_unigram_lexicon = read_nrc_hashtag_unigram(res.nrc_sentiment140_unigram_lexicon_path)
        self.nrc_sentiment140_bigram_lexicon = read_nrc_hashtag_bigram(res.nrc_sentiment140_bigram_lexicon_path)
        self.nrc_sentiment140_pair_lexicon = read_nrc_hashtag_pair(res.nrc_sentiment140_pair_lexicon_path)
        self.nrc_hashtag_sentimenthashtags_lexicon = read_nrc_hashtag_sentimenthashtags(res.nrc_hashtag_sentimenthashtags_path)

        self.mpqa_lexicon = read_mpqa(res.mpqa_lexicon_path)

        logger.info('Load carnegie clusters')
        self.carnegie_clusters = read_carnegie_clusters(res.carnegie_clusters_path)

    def build_pipeline(self):
        super().build_pipeline()
        self.text_features = [
            ['all caps', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['all caps', AF(feat.f_all_caps)]])],
            ['elongated', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['elongated', AF(feat.f_elongated_words)]])],
            ['emoticons', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['emoticons', AF(feat.F_Emoticons())]])],
            ['hashtags', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['hashtags', AF(feat.f_n_hashtags)]])],
            ['negation', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['negation', AF(feat.f_n_neg_context)]])],
            ['pos', Pipeline([
                ['selector', feat.ItemExtractor('pos')],
                ['vect', CountVectorizer()]])],
            ['punctuation', Pipeline([
                ['selector', feat.ItemExtractor('text')],
                ['punctuation', AF(feat.f_punctuation)]])],

            # Manually constructed lexicons
            ['nrc_emotion_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_emotion_lexicon))],
                ])],
            ['bing_liu_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.bing_liu_lexicon))],
            ])],
            ['mpqa_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.mpqa_lexicon))],
            ])],

            # Automatically constructed lexicons
            ['nrc_hashtag_unigram_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_hashtag_unigram_lexicon))],
                ])],
            ['nrc_hashtag_bigram_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_hashtag_bigram_lexicon, ngrams=2))],
                ])],
            ['nrc_hashtag_pair_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_hashtag_pair_lexicon, use_pair=True))],
                ## This feature really drop the perfs without normalization
                ['normalizer', Normalizer()],
            ])],
            ['nrc_sentiment140_unigram_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_sentiment140_unigram_lexicon))],
                ])],
            ['nrc_sentiment140_bigram_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_sentiment140_bigram_lexicon, ngrams=2))],
                ])],
            ['nrc_sentiment140_pair_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_sentiment140_pair_lexicon, use_pair=True))],
                ## This feature really drop the perfs without normalization
                ['normalizer', Normalizer()],
            ])],
            ['nrc_hashtag_sentimenthashtags_lexicon', Pipeline([
                ['selector', feat.ItemExtractor('tok_neg')],
                ['projection', AF(feat.F_NRC_Project_Lexicon(self.nrc_hashtag_sentimenthashtags_lexicon))],
            ])],
            ['clusters', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['clusters', AF(feat.IM_Project_Lexicon(self.carnegie_clusters))],
                ['convertion', CountVectorizer(binary=True)]])],
            ['word ngram', Pipeline([
                ['selector', feat.ItemExtractor('tok')],
                ['count', CountVectorizer(binary=True, lowercase=True,
                                          ngram_range=(1, 4))],
            ])],
            ['char ngram', Pipeline([
                ['selector', feat.ItemExtractor('text')],
                ['count', CountVectorizer(binary=True, analyzer='char', lowercase=True,
                                          ngram_range=(3, 5))],
            ])],
        ]

    def run_train(self):
        super().run_train()
        # clf = Pipeline([
        #     ('text_features', FeatureUnion(text_features)),
        #     #('standard scaler', StandardScaler(with_mean=False)),
        #     #('min/max scaler', MinMaxScaler()),
        #     ('max abs scaler', MaxAbsScaler()),
        #     ('clf', SVC(C=0.005, kernel='linear', max_iter=1000))]).fit(train.data, train.target)
        self.clf = Pipeline([
            ['text_features', FeatureUnion(self.text_features)],
            ['max abs scaler', MaxAbsScaler()],
            ['clf', SGDClassifier(loss='hinge',
                                  n_iter=100,
                                  n_jobs=5,
                                  # class_weight="balanced",
                                  # class_weight={0: 1, 1: 1, 2: 0.5},
                                  # class_weight={0: 1, 1: 1.5, 2: 0.25},
                                  random_state=42
            )]]).fit(self.train.data, self.train.target)

    def run_test(self):
        super().run_test()
        self.predicted = self.clf.predict(self.test.data)

    def print_results(self):
        super().print_results()
        logger.debug(pretty_pipeline(self.clf))
        logger.info('\n' +
                    metrics.classification_report(self.test.target, self.predicted,
                                                  target_names=self.test.labels))

        try:
            logger.info('\n' +
                        eval_with_semeval_script(self.test, self.predicted))
        except:
            pass

    def run(self):
        super().run()
        return self.clf, self.predicted, self.text_features
SVMRegister['NRCCanada'] = NRCCanada

KEEP_POS = ['FW',                                           # mots étrangers
            'RB', 'RBS', 'RBR',                             # adverbes
            'VBN', 'VBP', 'VBG', 'VB', 'VBD', 'VBZ', 'MD',  # verbes + modal
            'NNS', 'NNP', 'NNPS', 'NN',                     # nom
            'JJ', 'JJR',                                    # adjectifs
]


class Word2VecBase(NRCCanada):
    def __init__(self, topn=10000,
                 word2vec_param={},
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word2vec_param = {'size': 100,
                               'alpha': 0.025,
                               'window': 5,
                               'min_count': 5,
                               'max_vocab_size': None,
                               'sample': 0.001,
                               'seed': 1,
                               'workers': 3,
                               'min_alpha': 0.0001,
                               'sg': 0,
                               'hs': 0,
                               'negative': 5,
                               'cbow_mean': 1,
                               'hashfxn': hash,
                               'iter': 5,
                               'null_word': 0,
                               'trim_rule': None,
                               'sorted_vocab': 1,
                               'batch_words': 10000}
        self.word2vec_param.update(word2vec_param)
        self.topn = topn

    def build_pipeline_base(self):
        super().build_pipeline()
        self.text_features.append(
            ['word2vec', Pipeline([
            ['selector', feat.ItemExtractor('tok')],
            ['find closest', AF(feat.F_Find_Closest_In_Lexicon(self.word2vec,
                                                               self.bing_liu_lexicon,
                                                               self.topn))],
            ['convert to feature', AF(feat.Array_To_Feature('word2vec-closest'))],
            ['use feature', DictVectorizer()],
        ])])

    def build_pipeline_filtered(self):
        super().build_pipeline()
        self.text_features.append(
            ['word2vec', Pipeline([
            ['filter', AF(feat.KeepOn(keep_on='pos', keep=KEEP_POS))],
            ['selector', feat.ItemExtractor('tok')],
            ['find closest', AF(feat.F_Find_Closest_In_Lexicon(self.word2vec,
                                                               self.bing_liu_lexicon,
                                                               self.topn))],
            ['convert to feature', AF(feat.Array_To_Feature('word2vec-closest'))],
            ['use feature', DictVectorizer()],
        ])])

    def build_pipeline_filtered_mean(self):
        super().build_pipeline()
        self.text_features.append(
            ['word2vec', Pipeline([
            ['filter', AF(feat.KeepOn(keep_on='pos', keep=KEEP_POS))],
            ['selector', feat.ItemExtractor('tok')],
            ['find closest', AF(feat.F_Find_Closest_In_Lexicon(self.word2vec,
                                                               self.bing_liu_lexicon,
                                                               self.topn))],
            ['mean', feat.MeanVectors()],
        ])])
        
    build_pipeline = build_pipeline_filtered_mean


class WithSVD(Word2VecBase):
    def __init__(self, n_components=50, model_with_svd='word2vec',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_components = n_components
        self.model_with_svd = model_with_svd

    def build_pipeline(self):
        super().build_pipeline_filtered()
        el, idx = assoc_value(self.text_features, self.model_with_svd)
        el[1].steps.append(['SVD', TruncatedSVD(n_components=self.n_components)])


class Custom0(Word2VecBase):
    def load_resources(self):
        super().load_resources()
        self.word2vec = emb.get_custom0(word2vec_param=self.word2vec_param)
SVMRegister['Custom0'] = Custom0


class Custom0_with_SVD(Custom0, WithSVD):
    pass
SVMRegister['Custom0_with_SVD'] = Custom0_with_SVD


class GNews(Word2VecBase):
    def load_resources(self):
        super().load_resources()
        self.word2vec = emb.get_gnews()
SVMRegister['GNews'] = GNews


class GNews_with_SVD(GNews, WithSVD):
    pass
SVMRegister['GNews'] = GNews_with_SVD


class Custom1(Word2VecBase):
    def load_resources(self):
        super().load_resources()
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path)
        self.word2vec = emb.get_custom1(word2vec_param=self.word2vec_param,
                                        lexicon=self.bing_liu_lexicon)
SVMRegister['Custom1'] = Custom1


class Custom1_with_SVD(Custom1, WithSVD):
    pass
SVMRegister['Custom1_with_SVD'] = Custom1_with_SVD


class Custom2(Word2VecBase):
    def load_resources(self):
        super().load_resources()
        self.word2vec = emb.get_custom2()
SVMRegister['Custom1'] = Custom2


class Custom2_with_SVD(Custom2, WithSVD):
    pass
SVMRegister['Custom2_with_SVD'] = Custom2_with_SVD


class Custom3(Word2VecBase):
    def load_resources(self):
        super().load_resources()
        self.word2vec = emb.get_custom3()
SVMRegister['Custom1'] = Custom3


class Custom3_with_SVD(Custom3, WithSVD):
    pass
SVMRegister['Custom3_with_SVD'] = Custom3_with_SVD


class TestPipeline(SmallPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_truncate = None

    def build_pipeline(self):
        super().build_pipeline()
        self.my_set = set()
        self.text_features.append(['test', Pipeline([
            ['filter on', AF(feat.DropOn(drop_on='pos', drop=[]))],
            ['selector', feat.ItemExtractor('pos')],
            ['print', AF(lambda s: [[self.my_set.add(tag) for tag in s][0] or 1])],
        ])])

    def run_train(self):
        super().run_train()
        print(self.my_set)
