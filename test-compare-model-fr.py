#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


logger = logging.getLogger(__name__)

import numpy as np

import embeddings as emb
import resources as res
import reader
import utils

topn = 1000
sample_size = 5000
train_path = res.twitter_logger_fr_path

lidilem_adjectifs_lexicon = utils.remove_multi_words_in_lexicon(reader.read_lidilem_adjectifs(res.lidilem_adjectifs_lexicon_path))
lidilem_noms_lexicon = utils.remove_multi_words_in_lexicon(reader.read_lidilem_noms(res.lidilem_noms_lexicon_path))
lidilem_verbes_lexicon = utils.remove_multi_words_in_lexicon(reader.read_lidilem_verbes(res.lidilem_verbes_lexicon_path))
lidilem_lexicon = {}
# Merge lidilem lexicons
for l in [lidilem_adjectifs_lexicon, lidilem_noms_lexicon, lidilem_verbes_lexicon]:
    for w in l:
        lidilem_lexicon[w] = l[w]

blogoscopie_lexicon = utils.remove_multi_words_in_lexicon(reader.read_blogoscopie(res.blogoscopie_lexicon_path))
# Merge similar classes
for w in blogoscopie_lexicon:
    if blogoscopie_lexicon[w] in ['favorable', 'acceptation', 'accord-approx', 'accord-total']:
        blogoscopie_lexicon[w] = 'positif'
    if blogoscopie_lexicon[w] in ['defavorable', 'desaccord', 'rectificatif']:
        blogoscopie_lexicon[w] = 'negatif'

# anew_french = reader.read_anew(res.anew_french_lexicon_path)

# lexicons = [('lidilem_adjectifs', lidilem_adjectifs_lexicon),
#             ('lidilem_noms', lidilem_noms_lexicon),
#             ('lidilem_verbes', lidilem_verbes_lexicon),
#             ('blogoscopie', blogoscopie_lexicon)]

lexicons = [('lidilem', lidilem_lexicon),
            ('blogoscopie', blogoscopie_lexicon)]

for (lex_name, lex) in lexicons:
    utils.remove_multi_words_in_lexicon(lex)

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
