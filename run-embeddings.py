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

    # FileHandler
    fh = logging.FileHandler('log.txt', 'a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    logger.handlers = [sh, fh]

import embeddings as emb
import resources as res
import reader

bing_liu_lexicon = reader.read_bing_liu(res.bing_liu_lexicon_path)
nrc_emotion_lexicon = reader.read_nrc_emotion(res.nrc_emotion_lexicon_path)
word2vec_param = emb.default_word2vec_param
word2vec_param['workers'] = 6

# ~45min
# emb.get_custom0(word2vec_param=word2vec_param,
#                 force=True)
# ~55min
# emb.get_custom1(word2vec_param=word2vec_param, lexicon=bing_liu_lexicon,
#                 force=True, suffix='bing_liu')
# emb.get_custom1(word2vec_param=word2vec_param, lexicon=nrc_emotion_lexicon,
#                 force=True, suffix='nrc_emotion')

# ~70min
emb.get_custom_mce(word2vec_param=word2vec_param, lexicon=bing_liu_lexicon,
                   force=True, suffix='bing_liu')

emb.get_custom_mce(word2vec_param=word2vec_param, lexicon=nrc_emotion_lexicon,
                   force=True, suffix='nrc_emotion')
