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

models = []
# models.append(('model0', emb.get_custom0))
# models.append(('model1', emb.get_custom1))
models.append(('modelGnews', emb.get_gnews))

topn = 1000
sample_size = 5000

bing_liu_lexicon = reader.read_bing_liu(res.bing_liu_lexicon_path)
nrc_emotion_lexicon = reader.read_nrc_emotion(res.nrc_emotion_lexicon_path)
nrc_emotions_lexicon = reader.read_nrc_emotions(res.nrc_emotion_lexicon_path)
lexicons = [('bing', bing_liu_lexicon),
            ('nrc', nrc_emotion_lexicon),
            ('nrc2', nrc_emotions_lexicon)]

for m_name, m_fun in models:
    model = m_fun()
    for l_name, l_lex in lexicons:
        logger.info('Compare %s on %s', m_name, l_name)
        emb.compare_model_with_lexicon(model, l_lex)
        emb.compare_model_with_lexicon_class(model, l_lex)

        first = True
        for l_name2, l_lex2 in lexicons:
            if first:
                first = False
            else:
                model = m_fun()
            logger.info('Retrofit3 %s with %s', m_name, l_name2)
            model3 = emb.build_custom3(model, l_lex2)
            logger.info('Compare %s+3 on %s', m_name, l_name)
            emb.compare_model_with_lexicon(model3, l_lex)
            emb.compare_model_with_lexicon_class(model3, l_lex)

        for l_name2, l_lex2 in lexicons:
            model = m_fun()
            logger.info('Retrofit3_1 %s with %s', m_name, l_name2)
            model3 = emb.build_custom3_1(model, l_lex2)
            logger.info('Compare %s+3 on %s', m_name, l_name)
            emb.compare_model_with_lexicon(model3, l_lex)
            emb.compare_model_with_lexicon_class(model3, l_lex)


# for name1, lexicon1 in lexicons:
#     for m_name, model in models:
#         logger.info('Compare %s %s', m_name, name1)
#         emb.compare_model_with_lexicon(model, lexicon1, topn=topn,
#                                        sample_size=sample_size)
#         for name2, lexicon2 in lexicons:
#             logger.info('optimize3 %s with %s', m_name, name2)
#             model3 = emb.build_custom3(model, lexicon2)
#             emb.compare_model_with_lexicon(model3, lexicon1, topn=topn,
#                                            sample_size=sample_size)
#         for name2, lexicon2 in lexicons:
#             logger.info('optimize3_1 %s with %s', m_name, name2)
#             model3 = emb.build_custom3_1(model, lexicon2)
#             emb.compare_model_with_lexicon(model3, lexicon1, topn=topn,
#                                            sample_size=sample_size)

# Bulk testing
# for name1, lexicon1 in lexicons:
#     for m_name, model in models:
#         print('emb.compare_model_with_lexicon(%s, %s, topn=topn, sample_size=sample_size)' % (m_name, name1))
#         for name2, lexicon2 in lexicons:
#             print('model3 = emb.build_custom3(%s, %s)' % (m_name, name2))
#             print('emb.compare_model_with_lexicon(model3, %s, topn=topn, sample_size=sample_size)' % (name1))
#         for name2, lexicon2 in lexicons:
#             print('model3 = emb.build_custom3_1(%s, %s)' % (m_name, name2))
#             print('emb.compare_model_with_lexicon(model3, %s, topn=topn, sample_size=sample_size)' % (name1))

# for ratio in np.arange(0.1, 0.9, 0.1):
#     logger.info('split bing with ratio %f', ratio)
#     lex = lexicons[0][1]
#     train_l, test_l = utils.split_lexicon_train_test(lex)
#     model = emb.build_custom3(models[0][1], train_l)
#     logger.info('Compare with test_l')
#     emb.compare_model_with_lexicon(model, test_l, topn=topn, sample_size=sample_size)
#     logger.info('Compare with full lex')
#     emb.compare_model_with_lexicon(model, lex, topn=topn, sample_size=sample_size)
