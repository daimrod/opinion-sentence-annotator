#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pickle
import os

from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import read_semeval_dataset

from reader import Tokenizer
from reader import URLReplacer
from reader import UserNameReplacer

from utils import merge_classes
from utils import strings_to_integers

import features as feat

if 'logger' not in locals() and logging.getLogger('__run__') is not None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() %(levelname)-8s %(message)s')
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # FileHandler
    fh = logging.FileHandler('log.txt', 'a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


########## Pipeline
def preprocess(dataset_path, force=False, labels=['positive', 'negative', 'neutral']):
    preprocessed_path = dataset_path + '.pp.pickle'
    if not force and os.path.isfile(preprocessed_path):
        return preprocessed_path

    logger.info('Read dataset')
    logger.debug(dataset_path)
    dataset = read_semeval_dataset(dataset_path)
    logger.info('  Convert objective and neutral to objective/neutral')
    merge_classes(dataset.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    logger.info('  Build the target array')
    dataset.labels = labels
    target = strings_to_integers(dataset.target_names, labels)
    dataset.target.extend(target)

    logger.info('Normalize and tokenize the text')
    generator = (d['text'] for d in dataset.data)
    preprocessor = URLReplacer(generator)
    preprocessor = UserNameReplacer(preprocessor)
    preprocessor = Tokenizer(preprocessor, feat.happyfuntokenizer)
    for (d, tok) in zip(dataset.data, preprocessor):
        d['tok'] = tok

    logger.info('Extract Senna features')
    senna = feat.f_senna_multi([d['tok'] for d in dataset.data])
    for idx in range(len(dataset.data)):
        dataset.data[idx]['pos'] = senna[idx]['pos']
        dataset.data[idx]['chk'] = senna[idx]['chk']
        dataset.data[idx]['ner'] = senna[idx]['ner']

    logger.info('Identify negated contexts')
    for d in dataset.data:
        d['tok_neg'] = feat.im_neg_context(d['tok'])

    with open(preprocessed_path, 'wb') as p_file:
        pickle.dump(dataset, p_file)
    return preprocessed_path


class FullPipeline(object):
    def __init__(self, *args, **kwargs):
        logger.info('Init %s' % self.__class__.__name__)
        if len(args) != 0:
            logger.warning('args remaining %s' % args)
        if len(kwargs) != 0:
            logger.warning('kwargs remaining %s' % kwargs)

    def load_resources(self):
        logger.info('Load resources')

    def build_pipeline(self):
        logger.info('Build pipeline')

    def run_train(self):
        logger.info('Train')

    def run_test(self):
        logger.info('Test')

    def print_results(self):
        logger.info('Print results %s' % self.__class__.__name__)

    def run(self):
        logger.info('Run %s' % self.__class__.__name__)
        self.load_resources()
        self.build_pipeline()
        self.run_train()
        self.run_test()
        self.print_results()
