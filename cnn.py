#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import time

logger = logging.getLogger(__name__)

import embeddings as emb
from base import FullPipeline, preprocess
from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import read_bing_liu
from utils import eval_with_semeval_script
import lexicons

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
from keras.metrics import fbeta_score
from keras.metrics import precision
from keras.metrics import recall

from keras.callbacks import Callback

import numpy as np

CNNRegister = {}

# FIXME
def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall."""
    to_keep = y_pred[2] == 0
    y_pred = y_pred[to_keep]
    y_true = y_true[to_keep]
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return (2 * pre * rec) / 2


class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
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
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 previous_best=None,
                 mode='auto', period=1):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
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
            if 'acc' in self.monitor or 'fmeasure' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
        if previous_best is not None:
            self.best = previous_best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logger.warn('Can save best model only with %s available, '
                                'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            logger.info('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                        ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            logger.info('Epoch %05d: %s did not improve from %0.5f to %0.5f' %
                                        (epoch, self.monitor,
                                         self.best, current))
            else:
                if self.verbose > 0:
                    logger.info('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

class CNNBase(FullPipeline):
    def __init__(self,
                 train_truncate=0, test_truncate=0,
                 only_uid=None,
                 train_only_labels=['positive', 'negative', 'neutral'],
                 test_only_labels=['positive', 'negative', 'neutral'],
                 dev_only_labels=['positive', 'negative', 'neutral'],
                 repreprocess=False,
                 nb_epoch=2, batch_size=128,
                 nb_try=1, test_between_try=True,
                 max_sequence_length=1000,
                 shuffle=True,
                 max_nb_words=20000,
                 embedding_dim=100,
                 embedding_trainable=False,
                 metrics=['acc', 'fmeasure'],
                 monitor='val_fmeasure',
                 mode='auto',
                 best_model_path='best_model',
                 save_best_only=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_truncate = train_truncate
        self.test_truncate = test_truncate
        self.only_uid = only_uid
        self.train_only_labels = train_only_labels
        self.test_only_labels = test_only_labels
        self.dev_only_labels = dev_only_labels
        self.repreprocess = repreprocess
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.max_nb_words = max_nb_words
        self.embedding_dim = embedding_dim
        self.shuffle = shuffle
        self.embedding = None
        self.nb_try = nb_try
        self.test_between_try = test_between_try
        self.embedding_trainable = embedding_trainable
        self.metrics = metrics
        self.monitor = monitor
        self.mode = mode
        self.best_model_path_format = best_model_path
        self.best_model_path = None
        self.best_score = None
        self.save_best_only = save_best_only

    def load_fixed_embedding(self):
        logger.info('Preparing embedding matrix.')
        self.nb_words = min(self.max_nb_words, len(self.word_index))
        self.embeddings_index = {}
        self.embedding_dim = self.embedding.wv.syn0.shape[1]
        self.embedding_matrix = np.zeros((self.nb_words + 1,
                                          self.embedding_dim))
        for word, i in self.word_index.items():
            if i > self.max_nb_words:
                continue
            if word in self.embedding:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = self.embedding[word]

        # load pre-trained word embeddings into an Embedding layer
        self.embedding_layer = Embedding(self.nb_words + 1,
                                         self.embedding_dim,
                                         weights=[self.embedding_matrix],
                                         input_length=self.max_sequence_length)

    def load_trainable_embedding(self):
        logger.info('Preparing embedding matrix.')
        self.nb_words = min(self.max_nb_words, len(self.word_index))
        self.embedding_layer = Embedding(self.nb_words + 1,
                                         self.embedding_dim,
                                         input_length=self.max_sequence_length)
        # Embedding not pre-trained, force trainable
        self.embedding_trainable = True

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
        self.dev.filter_label(self.dev_only_labels)
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
        self.train = None

        self.dev_texts = [d['tok'] for d in self.dev.data]
        self.dev_sequences = self.tokenizer.texts_to_sequences(self.dev_texts)
        self.dev_data = pad_sequences(self.dev_sequences, maxlen=self.max_sequence_length)
        self.dev_labels = to_categorical(self.dev.target)
        self.dev_texts = None

        self.test_texts = [d['tok'] for d in self.test.data]
        self.test_sequences = self.tokenizer.texts_to_sequences(self.test_texts)
        self.test_data = pad_sequences(self.test_sequences, maxlen=self.max_sequence_length)
        self.test_labels = to_categorical(self.test.target)
        self.test_texts = None

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
        self.model.layers[1].trainable = self.embedding_trainable
        print('model built')
        print(self.model.summary())
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=self.metrics)

    def run_train(self):
        super().run_train()
        for j in range(self.nb_try):
            logger.info('Try number %d', j)
            self.build_pipeline()
            self.build_model()
            best_model_path = self.best_model_path_format + '_' + str(j)
            model_checkpoint = ModelCheckpoint(filepath=best_model_path,
                                               monitor=self.monitor,
                                               verbose=1,
                                               save_best_only=self.save_best_only,
                                               save_weights_only=True,
                                               mode=self.mode)
            if self.best_score is None:
                self.best_score = model_checkpoint.best
            self.hist = self.model.fit(self.train_data, [self.labels] * len(self.preds),
                                       validation_data=(self.dev_data,
                                                        self.dev_labels),
                                       nb_epoch=self.nb_epoch,
                                       batch_size=self.batch_size,
                                       verbose=1,
                                       callbacks=[model_checkpoint],
                                       shuffle=self.shuffle)
            if self.test_between_try:
                # Try with the current best model on test
                tmp = self.best_model_path
                self.best_model_path = best_model_path
                self.load_best_model()
                self.run_test()
                self.print_results()
                self.best_model_path = tmp

            # Test if the current (try) best model is the best of all time
            if self.best_score is None or model_checkpoint.monitor_op(model_checkpoint.best, self.best_score):
                logger.info('Try %05d: %s improved from %0.5f to %0.5f',
                            j, self.monitor, self.best_score, model_checkpoint.best)
                self.best_score = model_checkpoint.best
                self.best_model_path = best_model_path

        self.load_best_model()

    def load_best_model(self):
        del self.model
        self.build_pipeline()
        self.build_model()
        self.model.load_weights(self.best_model_path)

    def run_test(self):
        super().run_test()
        logger.info('Shape of data tensor: %s', self.test_data.shape)

        if len(self.preds) == 1:
            t_predicted = self.model.predict(self.test_data, verbose=1)
            if t_predicted.shape[-1] > 1:
                self.predicted = t_predicted.argmax(axis=-1)
            else:
                self.predicted = (t_predicted > 0.5).astype('int32')
        else:
            self.predicted = [None] * len(self.preds)
            for (i, t_predicted) in enumerate(self.model.predict(self.test_data, verbose=1)):
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
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        self.embedding = emb.get_custom1(lexicon=self.lexicon,
                                         suffix='bing_liu')
CNNRegister['CG_custom1'] = CNNChengGuo_Custom1


class CNNChengGuo_Custom3(CNNChengGuo):
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        model0 = emb.get_custom0()
        self.embedding = emb.build_custom3(model0,
                                           lexicon=self.lexicon)
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
        self.model.layers[1].trainable = self.embedding_trainable
        print('model built')
        print(self.model.summary())
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adadelta,
                           metrics=self.metrics)
CNNRegister['Rouvier_base'] = CNNRouvierBaseline


class CNNRouvierBaseline_custom0(CNNRouvierBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_custom0()
CNNRegister['Rouvier_base_custom0'] = CNNRouvierBaseline_custom0


class CNNRouvierBaseline_custom1(CNNRouvierBaseline):
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        self.embedding = emb.get_custom1(lexicon=self.lexicon,
                                         suffix=self.lexicon_name)
CNNRegister['Rouvier_base_custom1'] = CNNRouvierBaseline_custom1


class CNNRouvierBaseline_custom3(CNNRouvierBaseline):
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        model0 = emb.get_custom0()
        self.embedding = emb.build_custom3(model0,
                                           lexicon=self.lexicon)
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
        # dropout_hd2 = Dropout(self.dropout)(x)

        output1 = Dense(3, activation='softmax')(x)
        # output2 = Dense(3, activation='softmax')(dropout_hd2)

        self.preds = [output1]

    def build_model(self):
        self.model = Model(self.sequence_input, self.preds)
        self.model.layers[1].trainable = self.embedding_trainable
        print('model built')
        print(self.model.summary())
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adadelta,
                           metrics=self.metrics)
CNNRegister['Rouvier2016'] = CNNRouvier2016


class CNNRouvier2016_custom0(CNNRouvier2016):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_custom0()
CNNRegister['Rouvier2016_custom0'] = CNNRouvier2016_custom0


class CNNRouvier2016_custom1(CNNRouvier2016):
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        self.embedding = emb.get_custom1(lexicon=self.lexicon,
                                         suffix=self.lexicon_name)
CNNRegister['Rouvier2016_custom1'] = CNNRouvier2016_custom1


class CNNRouvier2016_custom_mce(CNNRouvier2016):
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        self.embedding = emb.get_custom_mce(lexicon=self.lexicon,
                                            suffix=self.lexicon_name)
CNNRegister['Rouvier2016_custom_mce'] = CNNRouvier2016_custom_mce


class CNNRouvier2016_custom3(CNNRouvier2016):
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        model0 = emb.get_custom0()
        self.embedding = emb.build_custom3(model0,
                                           lexicon=self.lexicon)
CNNRegister['Rouvier2016_custom3'] = CNNRouvier2016_custom3


class CNNRouvier2016_custom3_1(CNNRouvier2016):
    def __init__(self, lexicon_name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicon_name = lexicon_name
        self.lexicon = lexicons.get_lexicon(self.lexicon_name)
        model0 = emb.get_custom0()
        self.embedding = emb.build_custom3_1(model0,
                                             lexicon=self.lexicon)
CNNRegister['Rouvier2016_custom3_1'] = CNNRouvier2016_custom3_1


class CNNRouvier2016_gnews(CNNRouvier2016):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = emb.get_gnews()
CNNRegister['Rouvier2016_gnews'] = CNNRouvier2016_gnews
