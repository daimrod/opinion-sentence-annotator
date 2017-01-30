#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


if 'logger' not in locals():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() %(levelname)-8s %(message)s')
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.handlers = [sh]

import numpy as np

import embeddings as emb
import resources as res
import reader
import utils

topn = 1000
sample_size = 5000
train_path = res.twitter_logger_en_path

bing_liu_lexicon = reader.read_bing_liu(res.bing_liu_lexicon_path)
nrc_emotion_lexicon = reader.read_nrc_emotion(res.nrc_emotion_lexicon_path)
nrc_emotions_lexicon = reader.read_nrc_emotions(res.nrc_emotion_lexicon_path)
mpqa_lexicon = reader.read_mpqa(res.mpqa_lexicon_path)
lexicons = [('bing_liu', bing_liu_lexicon),
            ('nrc_polarity', nrc_emotion_lexicon),
            ('nrc_emotions', nrc_emotions_lexicon),
            ('mpqa', mpqa_lexicon)]

model_name = 'model0'
logger.info('Processing %s', model_name)
model = emb.get_custom0(train_path=train_path)
for (lex_test_name, lex_test) in lexicons:
    logger.info('Compare %s on %s', model_name, lex_test_name)
    emb.compare_model_with_lexicon(model, lex_test)
    emb.compare_model_with_lexicon_class(model, lex_test)

model_name = 'model1'
logger.info('Processing %s', model_name)
for lex_train_name, lex_train in lexicons:
    logger.info('Training %s with %s', model_name, lex_train_name)
    model = emb.get_custom1(train_path=train_path, lexicon=lex_train, suffix=lex_train_name)
    for lex_test_name, lex_test in lexicons:
        # Skip test lexicon if it's the same as the training lexicon
        if lex_test_name == lex_train_name:
            continue
        logger.info('Compare %s on %s', model_name, lex_test_name)
        emb.compare_model_with_lexicon(model, lex_test)
        emb.compare_model_with_lexicon_class(model, lex_test)


model_name = 'MCE'
logger.info('Processing %s', model_name)
for lex_train_name, lex_train in lexicons:
    logger.info('Training %s with %s', model_name, lex_train_name)
    model = emb.get_custom_mce(train_path=train_path, lexicon=lex_train, suffix=lex_train_name)
    for lex_test_name, lex_test in lexicons:
        # Skip test lexicon if it's the same as the training lexicon
        if lex_test_name == lex_train_name:
            continue
        logger.info('Compare %s on %s', model_name, lex_test_name)
        emb.compare_model_with_lexicon(model, lex_test)
        emb.compare_model_with_lexicon_class(model, lex_test)


model_name = 'faruqui'
logger.info('Processing %s', model_name)
for lex_train_name, lex_train in lexicons:
    logger.info('Training %s with %s', model_name, lex_train_name)
    model = emb.get_custom0(train_path=train_path)
    model = emb.build_custom3(model, lexicon=lex_train)
    for lex_test_name, lex_test in lexicons:
        # Skip test lexicon if it's the same as the training lexicon
        if lex_test_name == lex_train_name:
            continue
        logger.info('Compare %s on %s', model_name, lex_test_name)
        emb.compare_model_with_lexicon(model, lex_test)
        emb.compare_model_with_lexicon_class(model, lex_test)


model_name = 'faruqui++'
logger.info('Processing %s', model_name)
for lex_train_name, lex_train in lexicons:
    logger.info('Training %s with %s', model_name, lex_train_name)
    model = emb.get_custom0(train_path=train_path)
    model = emb.build_custom3_1(model, lexicon=lex_train)
    for lex_test_name, lex_test in lexicons:
        # Skip test lexicon if it's the same as the training lexicon
        if lex_test_name == lex_train_name:
            continue
        logger.info('Compare %s on %s', model_name, lex_test_name)
        emb.compare_model_with_lexicon(model, lex_test)
        emb.compare_model_with_lexicon_class(model, lex_test)
