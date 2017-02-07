#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


import numpy as np

import embeddings as emb
import resources as res
import reader
import utils

logger = logging.getLogger(__name__)

models = []
models.append(('model0', emb.get_custom0()))
models.append(('model1', emb.get_custom1()))
models.append(('modelGnews', emb.get_gnews()))

model0 = emb.get_custom0()
# model1 = emb.get_custom1()
modelGnews = emb.get_gnews()

topn = 1000
sample_size = 5000

bing_liu_lexicon = reader.read_bing_liu(res.bing_liu_lexicon_path)
nrc_emotion_lexicon = reader.read_nrc_emotion(res.nrc_emotion_lexicon_path)
nrc_emotions_lexicon = reader.read_nrc_emotions(res.nrc_emotion_lexicon_path)
lexicons = [('bing_liu_lexicon', bing_liu_lexicon),
            ('nrc_emotion_lexicon', nrc_emotion_lexicon),
            ('nrc_emotions_lexicon', nrc_emotions_lexicon)]

logger.info('bing_liu_lexicon')
logger.info('model0')
# emb.compare_model_with_lexicon(model0, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model0, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model0, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model0, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model0, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model0, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model0, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

logger.info('model1')
# emb.compare_model_with_lexicon(model1, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

logger.info('modelGnews')
# emb.compare_model_with_lexicon(modelGnews, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# modelGnews = emb.get_gnews()
# model3 = emb.build_custom3(modelGnews, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# modelGnews = emb.get_gnews()
# model3 = emb.build_custom3(modelGnews, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

# modelGnews = emb.get_gnews()
# model3 = emb.build_custom3(modelGnews, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, bing_liu_lexicon, topn=topn, sample_size=sample_size)

logger.info('NRC_EMOTION_LEXICON')
logger.info('model0')
emb.compare_model_with_lexicon(model0, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

model0 = emb.get_custom0()
model3 = emb.build_custom3(model0, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

model0 = emb.get_custom0()
model3 = emb.build_custom3(model0, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

model0 = emb.get_custom0()
model3 = emb.build_custom3(model0, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

model3 = emb.build_custom3_1(model0, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

model3 = emb.build_custom3_1(model0, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

model3 = emb.build_custom3_1(model0, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

logger.info('model1')
# emb.compare_model_with_lexicon(model1, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

logger.info('modelGnews')
emb.compare_model_with_lexicon(modelGnews, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3(modelGnews, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3(modelGnews, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3(modelGnews, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotion_lexicon, topn=topn, sample_size=sample_size)

logger.info('nrc_emotions_lexicon')
logger.info('model0')
emb.compare_model_with_lexicon(model0, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

model0 = emb.get_custom0()
model3 = emb.build_custom3(model0, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

model0 = emb.get_custom0()
model3 = emb.build_custom3(model0, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

model0 = emb.get_custom0()
model3 = emb.build_custom3(model0, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

model3 = emb.build_custom3_1(model0, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

model3 = emb.build_custom3_1(model0, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

model3 = emb.build_custom3_1(model0, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

logger.info('model1')
# emb.compare_model_with_lexicon(model1, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3(model1, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, bing_liu_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, nrc_emotion_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

# model3 = emb.build_custom3_1(model1, nrc_emotions_lexicon)
# emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

logger.info('modelGnews')
emb.compare_model_with_lexicon(modelGnews, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3(modelGnews, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3(modelGnews, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3(modelGnews, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, bing_liu_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, nrc_emotion_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)

modelGnews = emb.get_gnews()
model3 = emb.build_custom3_1(modelGnews, nrc_emotions_lexicon)
emb.compare_model_with_lexicon(model3, nrc_emotions_lexicon, topn=topn, sample_size=sample_size)
