#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.layers import Dense, Input
from keras.preprocessing.text import Tokenizer
from keras.models import Model

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

from svm import NRCCanada
from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
import reader as Reader
import resources as res
import features as feat

reader = Reader.TwitterLoggerTextReader(res.twitter_logger_en_path)
reader = Reader.URLReplacer(reader)
reader = Reader.UserNameReplacer(reader)
reader = Reader.Tokenizer(reader, feat.happyfuntokenizer)

X_train = reader
# y_train = 

nrc = NRCCanada()
nrc.load_resources()
X_test = [d['tok'] for d in nrc.test.data]
# y_test = 


# set parameters:
max_features = 5000
maxlen = 100  # cut texts after this number of words (among top
              # max_features most common words)
batch_size = 256
embedding_dims = 100
nb_filter = 250
filter_length = 50
hidden_dims = 250
nb_epoch = 50

tok = Tokenizer()
logger.info('PREPARE THE TEXT')
tok.fit_on_texts(X_train)
X_train = sequence.pad_sequences(tok.texts_to_sequences(X_train), maxlen)
X_test = sequence.pad_sequences(tok.texts_to_sequences(X_test), maxlen)

logger.info('ENCODER')
input_data = Input(shape=(maxlen,))
encoding_dim = int(maxlen/2)
encoded = Dense(encoding_dim, activation='relu')(input_data)

decoded = Dense(maxlen, activation='sigmoid')(encoded)

autoencoder = Model(input=input_data, output=decoded)

logger.info('DECODER')
encoded_input = Input(shape=(encoding_dim,))
decoder_layer =autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

logger.info('TRAIN & EVAL')
autoencoder.fit(X_train, X_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test))
logger.info('DONE')
