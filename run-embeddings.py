#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)

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
