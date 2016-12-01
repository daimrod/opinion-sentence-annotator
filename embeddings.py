#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import os

import gensim

from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import TwitterLoggerTextReader

from reader import Tokenizer
from reader import Splitter
from reader import LexiconProjecter
from reader import URLReplacer
from reader import UserNameReplacer

import features as feat
import resources as res

from base import FullPipeline

if 'logger' not in locals() and logging.getLogger('__run__') is not None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() %(levelname)-8s %(message)s')
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # FileHandler
    fh = logging.FileHandler('log.txt', 'a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    logger.handlers = [sh, fh]


class Custom0(FullPipeline):
    """This method adds a word2vec model projection to NRCCanada.
    """
    def load_custom0(self):
        logger.info('Load custom0 model')
        self.custom0_path = res.twitter_logger_en_path + '.word2vec'
        if os.path.exists(self.custom0_path) and os.path.getmtime(self.custom0_path) > os.path.getmtime(res.twitter_logger_en_path):
            self.custom0 = gensim.models.Word2Vec.load(self.custom0_path)
        else:
            reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
            reader = URLReplacer(reader)
            reader = UserNameReplacer(reader)
            reader = Tokenizer(reader, feat.happyfuntokenizer)
            reader = Splitter(reader)
            self.custom0 = gensim.models.Word2Vec(reader, **self.word2vec_param)
            self.custom0.init_sims(replace=True)
            self.custom0.save(self.custom0_path)


class Custom1(FullPipeline):
    def load_custom1(self):
        logger.info('Load custom1 model')
        self.custom1_path = res.twitter_logger_en_path + '.word2vec.custom1'
        if os.path.exists(self.custom1_path) and os.path.getmtime(self.custom1_path) > os.path.getmtime(res.twitter_logger_en_path):
            self.custom1 = gensim.models.Word2Vec.load(self.custom1_path)
        else:
            logger.info('Train custom1 model')
            reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
            reader = URLReplacer(reader)
            reader = UserNameReplacer(reader)
            reader = Tokenizer(reader, feat.happyfuntokenizer)
            reader = Splitter(reader)
            reader = LexiconProjecter(reader, self.bing_liu_lexicon)
            self.custom1 = gensim.models.Word2Vec(reader, **self.word2vec_param)
            self.custom1.init_sims(replace=True)
            self.custom1.save(self.custom1_path)


class Custom2(FullPipeline):
    def load_custom2(self):
        logger.info('Load custom2 model')
        self.custom2_path = '/home/jadi-g/src/thesis/SWE/demos/task1_wordsim/EmbedVector_TEXT8/semCOM1.Inter_run1.NEG0.0001/wordembed.semCOM1.dim100.win5.neg5.samp0.0001.inter0.hinge0.add0.decay0.l1.r1.embeded.txt'
        if os.path.exists(self.custom2_path) and os.path.getmtime(self.custom2_path) > os.path.getmtime(res.twitter_logger_en_path):
            self.custom2 = gensim.models.Word2Vec.load_word2vec_format(self.custom2_path, binary=False)
        else:
            logger.error('Custom2 model doesn\'t exist %s', self.custom2_path)
            raise ValueError


class Custom3(FullPipeline):
    def load_custom3(self):
        logger.info('Load custom3 model')
        self.custom3_path = '/tmp/word2vec.custom3.txt'
        if os.path.exists(self.custom3_path) and os.path.getmtime(self.custom3_path) > os.path.getmtime(res.twitter_logger_en_path):
            self.custom3 = gensim.models.Word2Vec.load_word2vec_format(self.custom3_path, binary=False)
        else:
            logger.error('Custom3 model doesn\'t exist %s', self.custom3_path)
            raise ValueError


class GNews(FullPipeline):
    """This method loads a gnews model.
    """
    def load_gnews(self):
        logger.info('Load gnews model')
        self.gnews_path = res.gnews_negative300_path
        if os.path.exists(self.gnews_path) and os.path.getmtime(self.gnews_path) > os.path.getmtime(res.twitter_logger_en_path):
            self.gnews = gensim.models.Word2Vec.load_word2vec_format(self.gnews_path, binary=True)
        else:
            logger.error('Gnews model doesn\'t exist %s', self.gnews_path)
            raise ValueError
