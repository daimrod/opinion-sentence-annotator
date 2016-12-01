#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import os

import gensim

from decorator import decorator

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


def make_get_model(build_function, name):
    def helper(train_path=res.twitter_logger_en_path,
               saved_model_path=res.twitter_logger_en_path + name,
               word2vec_param={},
               force=False):
        if not force and os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(train_path):
            model = gensim.models.Word2Vec.load(saved_model_path)
        else:
            model = build_function(train_path, saved_model_path, word2vec_param)
            model.init_sims(replace=True)
            model.save(saved_model_path)
        return model
    return helper


def build_custom0(train_path, saved_model_path, word2vec_param={}):
    """Build a Word2Vec model without any information.

    This is the 0 method, that is the baseline method."""
    logger.info('Build custom0 model')
    reader = TwitterLoggerTextReader(train_path)
    reader = URLReplacer(reader)
    reader = UserNameReplacer(reader)
    reader = Tokenizer(reader, feat.happyfuntokenizer)
    reader = Splitter(reader)
    return gensim.models.Word2Vec(reader, **word2vec_param)
get_custom0 = make_get_model(build_custom0, '.word2vec.custom0')


def build_custom1(train_path, saved_model_path,
                  lexicon,
                  word2vec_param={}):
    logger.info('Train custom1 model')
    reader = TwitterLoggerTextReader(train_path)
    reader = URLReplacer(reader)
    reader = UserNameReplacer(reader)
    reader = Tokenizer(reader, feat.happyfuntokenizer)
    reader = Splitter(reader)
    reader = LexiconProjecter(reader, lexicon)
    return gensim.models.Word2Vec(reader, **word2vec_param)
get_custom1 = make_get_model(build_custom1, '.word2vec.custom1')


def get_custom2():
    logger.info('Load custom2 model')
    saved_model_path = '/home/jadi-g/src/thesis/SWE/demos/task1_wordsim/EmbedVector_TEXT8/semCOM1.Inter_run1.NEG0.0001/wordembed.semCOM1.dim100.win5.neg5.samp0.0001.inter0.hinge0.add0.decay0.l1.r1.embeded.txt'
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path, binary=False)
    else:
        logger.error('Custom2 model doesn\'t exist %s', saved_model_path)
        raise ValueError


def get_custom3():
    logger.info('Load custom3 model')
    saved_model_path = '/tmp/word2vec.custom3.txt'
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path, binary=False)
    else:
        logger.error('Custom3 model doesn\'t exist %s', saved_model_path)
        raise ValueError


def get_gnews():
    logger.info('Load gnews model')
    saved_model_path = res.gnews_negative300_path
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path, binary=True)
    else:
        logger.error('Gnews model doesn\'t exist %s', saved_model_path)
        raise ValueError
