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
from base import FullPipeline, preprocess
from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import read_bing_liu
from utils import eval_with_semeval_script
from sklearn import metrics
import resources as res
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Convolution1D
from keras.layers import Dropout, merge
from keras.models import Model
from keras.models import K
from keras.optimizers import Adadelta
from keras.optimizers import SGD

from keras.callbacks import BaseLogger
from keras.callbacks import Callback

import numpy as np

CNNRegister = {}


class TestEpoch(BaseLogger):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        logger.info('epoch: %d', epoch)
        self.pipeline.run_test()
        self.pipeline.print_results()


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)


class SaveBestModel(Callback):
    '''
    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    '''
    def __init__(self, cnn_base, monitor='val_loss', verbose=0,
                 save_weights_only=False,
                 mode='auto', period=1):
        super().__init__()
        self.cnn_base = cnn_base
        self.monitor = monitor
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            logger.warn('ModelCheckpoint mode %s is unknown, '
                        'fallback to auto mode.' % (mode),
                        RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
        if self.cnn_base.best_score is not None:
            self.best = self.cnn_base.best_score

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            current = logs.get(self.monitor)
            if current is None:
                logger.warn('Can save best model only with %s available, '
                            'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model'
                              % (epoch, self.monitor, self.best, current))
                    self.best = current
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        self.cnn_base.best_model = self.model
        self.cnn_base.best_score = self.best


class CNNBase(FullPipeline):
    def __init__(self,
                 train_truncate=0, test_truncate=0,
                 only_uid=None,
                 train_only_labels=['positive', 'negative', 'neutral'],
                 test_only_labels=['positive', 'negative', 'neutral'],
                 repreprocess=False,
                 nb_epoch=2, batch_size=128,
                 nb_try=1,
                 max_sequence_length=1000,
                 shuffle=True,
                 max_nb_words=20000,
                 embedding_dim=100,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_truncate = train_truncate
        self.test_truncate = test_truncate
        self.only_uid = only_uid
        self.train_only_labels = train_only_labels
        self.test_only_labels = test_only_labels
        self.repreprocess = repreprocess
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.max_nb_words = max_nb_words
        self.embedding_dim = embedding_dim
        self.shuffle = shuffle
        self.embedding = None
        self.best_model = None
        self.best_score = None
        self.nb_try = nb_try

    def load_fixed_embedding(self):
        logger.info('Preparing embedding matrix.')
        self.nb_words = min(self.max_nb_words, len(self.word_index))
        self.embeddings_index = {}
        self.embedding_dim = self.embedding.syn0.shape[1]
        self.embedding_matrix = np.zeros((self.nb_words + 1,
                                          self.embedding_dim))
        for word, i in self.word_index.items():
            if i > self.max_nb_words:
                continue
            if word in self.embedding:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = self.embedding[word]

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        self.embedding_layer = Embedding(self.nb_words + 1,
                                         self.embedding_dim,
                                         weights=[self.embedding_matrix],
                                         input_length=self.max_sequence_length,
                                         trainable=False)

    def load_trainable_embedding(self):
        self.nb_words = min(self.max_nb_words, len(self.word_index))
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        self.embedding_layer = Embedding(self.nb_words + 1,
                                         self.embedding_dim,
                                         input_length=self.max_sequence_length,
                                         trainable=True)

    def load_resources(self):
        super().load_resources()
        logger.info('Load the corpus')
        with open(preprocess(res.train_path, force=self.repreprocess), 'rb') as p_file:
            self.train = pickle.load(p_file)
        with open(preprocess(res.test_path, force=self.repreprocess), 'rb') as p_file:
            self.test = pickle.load(p_file)

        with open(preprocess(res.dev_path, force=self.repreprocess), 'rb') as p_file:
            self.dev = pickle.load(p_file)

        self.train.truncate(self.train_truncate)
        self.test.truncate(self.test_truncate)
        self.train.filter_label(self.train_only_labels)
        self.test.filter_label(self.test_only_labels)
        self.dev.filter_label(self.test_only_labels)
        if self.only_uid is not None:
            self.test.filter_uid(self.only_uid)

        self.texts = [d['tok'] for d in self.train.data]
        self.labels_index = dict([(name, nid) for nid, name in enumerate(self.train.labels)])
        self.labels = to_categorical(self.train.target)
        logger.info('Found %s texts', len(self.texts))

        logger.info('Vectorize the text samples into a 2D integer tensor')
        self.tokenizer = Tokenizer(nb_words=self.max_nb_words)
        self.tokenizer.fit_on_texts(self.texts)
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)

        self.dev_texts = [d['tok'] for d in self.dev.data]
        self.dev_sequences = self.tokenizer.texts_to_sequences(self.dev_texts)
        self.dev_data = pad_sequences(self.dev_sequences, maxlen=self.max_sequence_length)

        self.word_index = self.tokenizer.word_index
        logger.info('Found %s unique tokens.', len(self.word_index))

        self.train_data = pad_sequences(self.sequences, maxlen=self.max_sequence_length)

        logger.info('Shape of data tensor: %s', self.train_data.shape)
        logger.info('Shape of label tensor: %s', self.labels.shape)
        logger.info('label index: %s', self.labels_index)

        if self.embedding is None:
            self.load_trainable_embedding()
        else:
            self.load_fixed_embedding()

    def build_pipeline(self):
        super().build_pipeline()
        self.sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        self.embedded_sequences = self.embedding_layer(self.sequence_input)
        x = Conv1D(128, 5, activation='relu')(self.embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        self.preds = [Dense(len(self.labels_index), activation='softmax')(x)]

    def build_model(self):
        self.model = Model(self.sequence_input, self.preds)
        print('model built')
        print(self.model.summary())
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=[fmeasure])

    def run_train(self):
        super().run_train()
        for j in range(self.nb_try):
            self.build_model()

            self.model.fit(self.train_data, [self.labels] * len(self.preds),
                           validation_data=(self.dev_data,
                                            to_categorical(self.dev.target)),
                           nb_epoch=self.nb_epoch,
                           batch_size=self.batch_size,
                           verbose=1,
                           callbacks=[SaveBestModel(self,
                                                    monitor='val_fmeasure',
                                                    mode='max')],
                           shuffle=self.shuffle)

    def run_test(self):
        super().run_test()
        self.test_texts = [d['tok'] for d in self.test.data]
        self.test_sequences = self.tokenizer.texts_to_sequences(self.test_texts)
        self.test_data = pad_sequences(self.test_sequences, maxlen=self.max_sequence_length)
        logger.info('Shape of data tensor: %s', self.test_data.shape)

        if len(self.preds) == 1:
            t_predicted = self.best_model.predict(self.test_data, verbose=1)
            if t_predicted.shape[-1] > 1:
                self.predicted = t_predicted.argmax(axis=-1)
            else:
                self.predicted = (t_predicted > 0.5).astype('int32')
        else:
            self.predicted = [None] * len(self.preds)
            for (i, t_predicted) in enumerate(self.best_model.predict(self.test_data, verbose=1)):
                if t_predicted.shape[-1] > 1:
                    self.predicted[i] = t_predicted.argmax(axis=-1)
                else:
                    self.predicted[i] = (t_predicted > 0.5).astype('int32')

    def print_results(self):
        super().print_results()
        if len(self.preds) == 1:
            logger.info('\n' +
                        metrics.classification_report(self.test.target, self.predicted,
                                                      target_names=self.test.labels))
            try:
                logger.info('\n' +
                            eval_with_semeval_script(self.test, self.predicted))
            except:
                pass
        else:
            for (i, predicted) in enumerate(self.predicted):
                logger.info('\n' +
                            'Output number : %d\n' % i +
                            metrics.classification_report(self.test.target, predicted,
                                                          target_names=self.test.labels))

                try:
                    logger.info('\n' +
                                eval_with_semeval_script(self.test, predicted))
                except:
                    pass
CNNRegister['CNNBase'] = CNNBase


class CNNChengGuo(CNNBase):
    """More or less c/c from https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
An Keras implementation of Cheng Guao CNN for sentence classification.

I add minor adjustments to make it work for the Semeval Sentiment Analsysis tasks.
    """
    def __init__(self, ngram_filters=[3, 4, 5], nb_filter=100, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ngram_filters = ngram_filters
        self.nb_filter = nb_filter
        self.dropout = dropout

    def build_pipeline(self):
        super().build_pipeline()
        # again, credit to Cheng Guo

        self.sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        self.embedded_sequences = self.embedding_layer(self.sequence_input)
        x = Dropout(self.dropout)(self.embedded_sequences)
        ngram_filters = []
        for n_gram in self.ngram_filters:
            x1 = Convolution1D(nb_filter=self.nb_filter,
                               filter_length=n_gram,
                               border_mode='valid',
                               activation='relu',
                               subsample_length=1)(x)
            x1 = MaxPooling1D(pool_length=self.max_sequence_length - n_gram + 1)(x1)
            x1 = Flatten()(x1)
            ngram_filters.append(x1)
        x = merge(ngram_filters, mode='concat')
        x = Dropout(self.dropout)(x)
        self.preds = [Dense(len(self.labels_index), activation='sigmoid')(x)]
CNNRegister['CG'] = CNNChengGuo


class CNNChengGuo_Custom0(CNNChengGuo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_custom0()
CNNRegister['CG_custom0'] = CNNChengGuo_Custom0


class CNNChengGuo_Custom1(CNNChengGuo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                              res.bing_liu_lexicon_path['positive'])
        self.embedding = emb.get_custom1(lexicon=self.bing_liu_lexicon)
CNNRegister['CG_custom1'] = CNNChengGuo_Custom1


class CNNChengGuo_Custom3(CNNChengGuo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                              res.bing_liu_lexicon_path['positive'])
        model0 = emb.get_custom0()
        self.embedding = emb.build_custom3(model0,
                                           lexicon=self.bing_liu_lexicon)
CNNRegister['CG_custom3'] = CNNChengGuo_Custom3


class CNNChengGuo_Gnews(CNNChengGuo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_gnews()
CNNRegister['CG_gnews'] = CNNChengGuo_Gnews


class CNNRouvierBaseline(CNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.embedding_dim = 100
        self.max_nb_words = 50000
        self.max_sequence_length = 100
        self.nb_filter = 500
        self.ngram_filters = [1, 2, 3, 4, 5]
        self.dropout = 0.4

    def build_pipeline(self):
        self.sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        self.embedded_sequences = self.embedding_layer(self.sequence_input)
        x = self.embedded_sequences
        ngram_filters = []
        for n_gram in self.ngram_filters:
            x1 = Convolution1D(nb_filter=self.nb_filter,
                               filter_length=n_gram,
                               border_mode='valid',
                               activation='relu')(x)
            x1 = MaxPooling1D(pool_length=self.max_sequence_length - n_gram + 1,
                              stride=None,
                              border_mode='valid')(x1)
            x1 = Flatten()(x1)
            ngram_filters.append(x1)
        x = merge(ngram_filters, mode='concat')
        x = Dropout(self.dropout)(x)

        self.preds = [Dense(len(self.labels_index), activation='softmax')(x)]

    def build_model(self):
        self.model = Model(self.sequence_input, self.preds)
        print('model built')
        print(self.model.summary())
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adadelta,
                           metrics=[fmeasure])
CNNRegister['Rouvier_base'] = CNNRouvierBaseline


class CNNRouvierBaseline_custom0(CNNRouvierBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_custom0()
CNNRegister['Rouvier_base_custom0'] = CNNRouvierBaseline_custom0


class CNNRouvierBaseline_custom1(CNNRouvierBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                              res.bing_liu_lexicon_path['positive'])
        self.embedding = emb.get_custom1(lexicon=self.bing_liu_lexicon)
CNNRegister['Rouvier_base_custom1'] = CNNRouvierBaseline_custom1


class CNNRouvierBaseline_custom3(CNNRouvierBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                              res.bing_liu_lexicon_path['positive'])
        model0 = emb.get_custom0()
        self.embedding = emb.build_custom3(model0,
                                           lexicon=self.bing_liu_lexicon)
CNNRegister['Rouvier_base_custom3'] = CNNRouvierBaseline_custom3


class CNNRouvierBaseline_gnews(CNNRouvierBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_gnews()
CNNRegister['Rouvier_base_gnews'] = CNNRouvierBaseline_gnews


class CNNRouvier2016(CNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.embedding_dim = 100
        self.max_nb_words = 50000
        self.max_sequence_length = 100
        self.nb_filter = 500
        self.ngram_filters = [1, 2, 3, 4, 5]
        self.ngram_filters_extended = [1, 2, 3, 4, 5, 6]
        self.dropout = 0.4

    def build_pipeline(self):
        self.sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        self.embedded_sequences = self.embedding_layer(self.sequence_input)
        model_input = self.embedded_sequences
        output3 = Flatten()(model_input)
        output3 = Dense(200, activation='relu')(output3)
        output3 = Dropout(self.dropout)(output3)
        output3 = Dense(3, activation='softmax')(output3)

        ngram_filters = []
        for n_gram in self.ngram_filters:
            x1 = Convolution1D(nb_filter=self.nb_filter,
                               filter_length=n_gram,
                               border_mode='valid',
                               activation='relu')(model_input)
            x1 = MaxPooling1D(pool_length=self.max_sequence_length - n_gram + 1,
                              stride=None)(x1)
            x1 = Flatten()(x1)
            ngram_filters.append(x1)

        x = merge(ngram_filters, mode='concat')
        dropout = Dropout(self.dropout)(x)

        x = Dense(512, activation='relu')(dropout)
        dropout_hd1 = Dropout(self.dropout)(x)

        x = Dense(512, activation='relu')(dropout_hd1)
        dropout_hd2 = Dropout(self.dropout)(x)

        output1 = Dense(3, activation='softmax')(x)
        output2 = Dense(3, activation='softmax')(dropout_hd2)

        self.preds = [output1]

    def build_model(self):
        self.model = Model(self.sequence_input, self.preds)
        print('model built')
        print(self.model.summary())
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adadelta,
                           metrics=[fmeasure])
CNNRegister['Rouvier2016'] = CNNRouvier2016


class CNNRouvier2016_custom0(CNNRouvier2016):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_custom0()
CNNRegister['Rouvier2016_custom0'] = CNNRouvier2016_custom0


class CNNRouvier2016_custom1(CNNRouvier2016):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                              res.bing_liu_lexicon_path['positive'])
        self.embedding = emb.get_custom1(lexicon=self.bing_liu_lexicon)
CNNRegister['Rouvier2016_custom1'] = CNNRouvier2016_custom1


class CNNRouvier2016_custom3(CNNRouvier2016):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bing_liu_lexicon = read_bing_liu(res.bing_liu_lexicon_path['negative'],
                                              res.bing_liu_lexicon_path['positive'])
        model0 = emb.get_custom0()
        self.embedding = emb.build_custom3(model0,
                                           lexicon=self.bing_liu_lexicon)
CNNRegister['Rouvier2016_custom3'] = CNNRouvier2016_custom3


class CNNRouvier2016_gnews(CNNRouvier2016):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_gnews()
CNNRegister['Rouvier2016_gnews'] = CNNRouvier2016_gnews
