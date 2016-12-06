#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import os
from subprocess import Popen, PIPE

from collections import Counter
from collections import OrderedDict

import gensim

from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import TwitterLoggerTextReader

from reader import Tokenizer
from reader import Splitter
from reader import LexiconProjecter
from reader import URLReplacer
from reader import UserNameReplacer
from reader import LineReader

import features as feat
import resources as res

import tempfile

default_word2vec_param = {
    # sg defines the training algorithm. By default (sg=0), CBOW is
    # used. Otherwise (sg=1), skip-gram is employed.
    'sg': 0,

    # size is the dimensionality of the feature vectors.
    'size': 100,

    # window is the maximum distance between the current and predicted
    # word within a sentence.
    'window': 5,

    # alpha is the initial learning rate (will linearly drop to
    # min_alpha as training progresses).
    'alpha': 0.025,

    # seed = for the random number generator. Initial vectors for each
    # word are seeded with a hash of the concatenation of word +
    # str(seed). Note that for a fully deterministically-reproducible
    # run, you must also limit the model to a single worker thread, to
    # eliminate ordering jitter from OS thread scheduling. (In Python
    # 3, reproducibility between interpreter launches also requires
    # use of the PYTHONHASHSEED environment variable to control hash
    # randomization.)
    'seed': 1,

    # min_count = ignore all words with total frequency lower than
    # this.
    'min_count': 5,

    # max_vocab_size = limit RAM during vocabulary building; if there
    # are more unique words than this, then prune the infrequent ones.
    # Every 10 million word types need about 1GB of RAM. Set to None
    # for no limit (default).
    'max_vocab_size': None,

    # sample = threshold for configuring which higher-frequency words
    # are randomly downsampled; default is 1e-3, useful range is (0,
    # 1e-5).
    'sample': 0.001,

    # workers = use this many worker threads to train the model
    # (=faster training with multicore machines).
    'workers': 3,

    # hs = if 1, hierarchical softmax will be used for model training.
    # If set to 0 (default), and negative is non-zero, negative
    # sampling will be used.
    'hs': 0,

    # negative = if > 0, negative sampling will be used, the int for
    # negative specifies how many “noise words” should be drawn
    # (usually between 5-20). Default is 5. If set to 0, no negative
    # samping is used.
    'negative': 5,

    # cbow_mean = if 0, use the sum of the context word vectors. If 1
    # (default), use the mean. Only applies when cbow is used.
    'cbow_mean': 1,

    # hashfxn = hash function to use to randomly initialize weights,
    # for increased training reproducibility. Default is Python’s
    # rudimentary built in hash function.
    'hashfxn': hash,

    # iter = number of iterations (epochs) over the corpus. Default is 5.
    'iter': 5,

    # trim_rule = vocabulary trimming rule, specifies whether certain
    # words should remain in the vocabulary, be trimmed away, or
    # handled using the default (discard if word count < min_count).
    # Can be None (min_count will be used), or a callable that accepts
    # parameters (word, count, min_count) and returns either
    # utils.RULE_DISCARD, utils.RULE_KEEP or utils.RULE_DEFAULT. Note:
    # The rule, if given, is only used prune vocabulary during
    # build_vocab() and is not stored as part of the model.
    'trim_rule': None,

    # sorted_vocab = if 1 (default), sort the vocabulary by descending
    # frequency before assigning word indexes.
    'sorted_vocab': 1,

    # batch_words = target size (in words) for batches of examples
    # passed to worker threads (and thus cython routines). Default is
    # 10000. (Larger batches will be passed if individual texts are
    # longer than 10000 words, but the standard cython code truncates
    # to that maximum.)
    'batch_words': 10000,
}

if 'logger' not in locals() and logging.getLogger('__run__') is not None:
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


def make_get_model(build_function, name):
    def helper(train_path=res.twitter_logger_en_path,
               saved_model_path=res.twitter_logger_en_path + name,
               word2vec_param=default_word2vec_param,
               force=False,
               **kwargs):
        if not force and os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(train_path):
            model = gensim.models.Word2Vec.load(saved_model_path)
        else:
            model = build_function(train_path, word2vec_param=word2vec_param, **kwargs)
            model.init_sims(replace=True)
            model.save(saved_model_path)
        return model
    return helper


def build_custom0(train_path, word2vec_param=default_word2vec_param):
    """Build a Word2Vec model without any information.

    This is the 0 method, that is the baseline method."""
    logger.info('Build custom0 model')
    source = TwitterLoggerTextReader(train_path)
    source = URLReplacer(source)
    source = UserNameReplacer(source)
    source = Tokenizer(source, feat.happyfuntokenizer)
    source = Splitter(source)
    return gensim.models.Word2Vec(source, **word2vec_param)
get_custom0 = make_get_model(build_custom0, '.word2vec.custom0')


def build_custom1(train_path,
                  word2vec_param=default_word2vec_param,
                  lexicon=None):
    if lexicon is None:
        raise ValueError('Empty lexicon')
    logger.info('Train custom1 model')
    source = TwitterLoggerTextReader(train_path)
    source = URLReplacer(source)
    source = UserNameReplacer(source)
    source = Tokenizer(source, feat.happyfuntokenizer)
    source = Splitter(source)
    source = LexiconProjecter(source, lexicon)
    return gensim.models.Word2Vec(source, **word2vec_param)
get_custom1 = make_get_model(build_custom1, '.word2vec.custom1')


def get_custom2():
    logger.info('Load custom2 model')
    saved_model_path = '/home/jadi-g/src/thesis/SWE/demos/task1_wordsim/EmbedVector_TEXT8/semCOM1.Inter_run1.NEG0.0001/wordembed.semCOM1.dim100.win5.neg5.samp0.0001.inter0.hinge0.add0.decay0.l1.r1.embeded.txt'
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path, binary=False)
    else:
        logger.error('Custom2 model doesn\'t exist %s', saved_model_path)
        raise ValueError


def build_custom2(train_path,
                  word2vec_param=default_word2vec_param,
                  lexicon=None):
    """Build a Word2Vec model using SWE method (optimization with
inequalities)."""
    if lexicon is None:
        raise ValueError('Empty lexicon')
    # The train corpus in a file
    # The vocab in a file
    # Some parameters
    # The path to SWE_Train exec
    model = None
    source = LineReader(train_path)
    source = URLReplacer(source)
    source = UserNameReplacer(source)
    source = Tokenizer(source, feat.happyfuntokenizer)
    source = Splitter(source)
    input_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
    output_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
    vocab_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
    vocab = Counter()
    output_file.close()
    for line in source:
        vocab.update(line)
        input_file.write(' '.join(line))
        input_file.write('\n')
    vocab = OrderedDict(sorted(vocab.items(), key=lambda t: t[1],
                               reverse=True))
    for word in vocab:
        vocab_file.write('%s\t%d\n' % (word, vocab[word]))
    vocab_file.close()

    input_file.close()
    p = Popen(['bin/SWE_Train',
               '-train', train_path,
               '-read-vocab', vocab_file.name,
               '-output', output_file.name,
               '-size', str(word2vec_param['size']),
               '-window', str(word2vec_param['window']),
               '-sample', str(word2vec_param['sample']),
               '-hs', str(word2vec_param['hs']),
               '-iter', str(word2vec_param['iter']),
               '-min-count', str(word2vec_param['min_count']),
               '-sem-train', os.path.join(res.SWE_PATH, 'semantics/TEXT8/SemWE.EN.KnowDB.COM1.inTEXT8.train'),
               '-sem-valid', os.path.join(res.SWE_PATH, 'semantics/TEXT8/SemWE.EN.KnowDB.COM1.inTEXT8.valid')],
              stdin=PIPE,
              stdout=PIPE,
              stderr=PIPE,
              cwd=res.SWE_PATH)
    out, err = p.communicate()
    err = err.decode()
    out = out.decode()
    if p.returncode != 0:
        logger.error(out)
        logger.error(err)
    else:
        logger.info(out)
        model = gensim.models.Word2Vec.load_word2vec_format(output_file.name,
                                                            binary=False)
    os.remove(vocab_file.name)
    os.remove(input_file.name)
    os.remove(output_file.name)
    return model


def get_custom3():
    logger.info('Load custom3 model')
    saved_model_path = '/tmp/word2vec.custom3.txt'
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path, binary=False)
    else:
        logger.error('Custom3 model doesn\'t exist %s', saved_model_path)
        raise ValueError


def get_gnews():
    logger.info('Load gnews model')
    saved_model_path = res.gnews_negative300_path
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path, binary=True)
    else:
        logger.error('Gnews model doesn\'t exist %s', saved_model_path)
        raise ValueError
