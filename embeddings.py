#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import os
from subprocess import Popen, PIPE
import tempfile
import codecs
import re
import random

from collections import Counter
from collections import OrderedDict
from collections import defaultdict

import gensim
import numpy as np
from copy import deepcopy

from reader import Dataset  # We need this import because we're loading
                            # a Dataset with pickle
from reader import TwitterLoggerTextReader

from reader import Tokenizer
from reader import Splitter
from reader import LineReader
from reader import LexiconProjecter
from reader import GenericTextReader
from reader import w_norm

import features as feat
import resources as res
import utils

default_word2vec_param = {
    # sg defines the training algorithm. By default (sg=0), CBOW is
    # used. Otherwise (sg=1), skip-gram is employed.
    'sg': 0,

    # size is the dimensionality of the feature vectors.
    'size': 300,

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
    'min_count': 20,

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
               force=False, suffix='',
               **kwargs):
        saved_model_path = '%s_size_%d_window_%d_%s' % (saved_model_path,
                                                     word2vec_param['size'],
                                                        word2vec_param['window'],
                                                        suffix)
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
    source = GenericTextReader(source)
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
    source = GenericTextReader(source)
    source = Splitter(source)
    source = LexiconProjecter(source, lexicon)
    return gensim.models.Word2Vec(source, **word2vec_param)
get_custom1 = make_get_model(build_custom1, '.word2vec.custom1')


def get_custom2():
    logger.info('Load custom2 model')
    saved_model_path = '/home/jadi-g/src/thesis/SWE/demos/task1_wordsim/EmbedVector_TEXT8/semCOM1.Inter_run1.NEG0.0001/wordembed.semCOM1.dim100.win5.neg5.samp0.0001.inter0.hinge0.add0.decay0.l1.r1.embeded.txt'
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path,
                                                           binary=True,
                                                           unicode_errors='replace')
    else:
        logger.error('Custom2 model doesn\'t exist %s', saved_model_path)
        raise ValueError


def build_custom2(train_path,
                  word2vec_param=default_word2vec_param,
                  lexicon=None, valid_num=0.1, top=10,
                  clean_after=True):
    """Build a Word2Vec model using SWE method (optimization with
inequalities).

    Args:
        lexicon: The lexicon used to build the inequalities.
        valid_num: How much inequations should be used for cross-validation (either a floar between 0 and 1 or an integer.
        top: See feat.build_ineq_for_model
        clean_after: Clean the files after building the model if True."""
    if lexicon is None:
        raise ValueError('Empty lexicon')
    model = None
    source = TwitterLoggerTextReader(train_path)
    source = GenericTextReader(source, lower=True)
    source = Splitter(source)

    input_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                             delete=False, prefix='input')
    output_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                              delete=False, prefix='output')
    output_file.close()

    ineq_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                            delete=False, prefix='ineq')
    ineq_file.close()

    vocab_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                             delete=False, prefix='vocab')

    try:
        logger.info('Build vocabulary file')
        vocab = Counter()
        for line in source:
            vocab.update(line)
            input_file.write(' '.join(line))
            input_file.write('\n')
        vocab = OrderedDict(sorted(vocab.items(), key=lambda t: t[1],
                                   reverse=True))
        for word in vocab:
            # Ignore word with freq < min_count
            if 'min_count' in word2vec_param and vocab[word] < word2vec_param['min_count']:
                del vocab[word]
                continue
            vocab_file.write('%s\t%d\n' % (word, vocab[word]))
        vocab_file.close()

        model0 = get_custom0(word2vec_param=word2vec_param)
        new_lexicon = {}
        for w in lexicon:
            # Ignore word not in vocab nor in model
            if w not in model0 or w not in vocab:
                continue
            new_lexicon[w] = lexicon[w]
        feat.build_ineq_for_model(model0, new_lexicon,
                                  output_path=ineq_file.name,
                                  vocab=list(vocab),
                                  top=top)
        utils.split_train_valid(ineq_file.name, valid_num=valid_num)

        input_file.close()
        cmd = ['bin/SWE_Train',
               '-train', input_file.name,
               '-read-vocab', vocab_file.name,
               '-output', output_file.name,
               '-size', str(word2vec_param['size']),
               '-window', str(word2vec_param['window']),
               '-sample', str(word2vec_param['sample']),
               '-hs', str(word2vec_param['hs']),
               '-iter', str(word2vec_param['iter']),
               '-min-count', str(word2vec_param['min_count']),
               '-sem-train', ineq_file.name + '.train',
               '-sem-valid', ineq_file.name + '.valid',
        ]
        logger.info(' '.join(cmd))
        p = Popen(cmd,
                  stdin=PIPE,
                  stdout=PIPE,
                  stderr=PIPE,
                  cwd=res.SWE_PATH)
        out, err = p.communicate()
        err = err.decode()
        out = out.decode()
        logger.info(out)
        logger.error(err)
        if p.returncode == 0:
            model = gensim.models.Word2Vec.load_word2vec_format(output_file.name,
                                                                binary=False,
                                                                unicode_errors='replace')
    finally:
        if clean_after:
            os.remove(vocab_file.name)
            os.remove(input_file.name)
            os.remove(output_file.name)
            os.remove(ineq_file.name)
            os.remove(ineq_file.name + '.train')
            os.remove(ineq_file.name + '.valid')
    return model


def build_custom_mce(train_path,
                  word2vec_param=default_word2vec_param,
                  lexicon=None, valid_num=0.1, top=10,
                  clean_after=True):
    """Build a Word2Vec model using MCE method.

    Args:
        lexicon: The lexicon used to build the inequalities.
        valid_num: How much inequations should be used for cross-validation (either a floar between 0 and 1 or an integer.
        top: See feat.build_ineq_for_model
        clean_after: Clean the files after building the model if True."""
    if lexicon is None:
        raise ValueError('Empty lexicon')

    source = TwitterLoggerTextReader(train_path)
    source = GenericTextReader(source, lower=True)
    source = Splitter(source)

    input_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                             delete=False, prefix='input')
    for line in source:
        input_file.write(' '.join(line))
        input_file.write('\n')
    input_file.close()

    output_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                              delete=False, prefix='output')
    output_file.close()

    syn_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                           delete=False, prefix='syn')
    ant_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
                                           delete=False, prefix='ant')

    lexicon_inv = utils.invert_dict_nonunique(lexicon)
    # Generate syn_file
    for c in lexicon_inv:
        syn_file.write('\t'.join(lexicon_inv[c]))
        syn_file.write('\n')
    syn_file.close()

    # Generate ant_file
    for cur_c in lexicon_inv:
        for word in lexicon_inv[cur_c]:
            for c in lexicon_inv:
                # skip current observed class
                if c == cur_c:
                    continue
                ant_file.write(word + '\t' + '\t'.join(lexicon_inv[c]))
                ant_file.write('\n')
    ant_file.close()

    cmd = ['./word2vec',
           '-train', input_file.name,
           '-output', output_file.name,
           '-size', str(word2vec_param['size']),
           '-window', str(word2vec_param['window']),
           '-sample', str(word2vec_param['sample']),
           '-hs', str(word2vec_param['hs']),
           '-iter', str(word2vec_param['iter']),
           '-min-count', str(word2vec_param['min_count']),
           '-read-syn', syn_file.name,
           '-read-ant', ant_file.name,
    ]
    logger.info(' '.join(cmd))
    p = Popen(cmd,
              stdin=PIPE,
              stdout=PIPE,
              stderr=PIPE,
              cwd=res.MCE_PATH)
    out, err = p.communicate()
    err = err.decode()
    out = out.decode()
    logger.info(out)
    logger.error(err)
    if p.returncode == 0:
        model = gensim.models.Word2Vec.load_word2vec_format(output_file.name,
                                                            binary=False,
                                                            unicode_errors='replace')
    if clean_after:
        os.remove(output_file.name)
        os.remove(input_file.name)
        os.remove(syn_file.name)
        os.remove(ant_file.name)
    return model
get_custom_mce = make_get_model(build_custom_mce, '.word2vec.custom_mce')


def old_get_custom3():
    logger.info('Load custom3 model')
    saved_model_path = '/tmp/word2vec.custom3.txt'
    if (os.path.exists(saved_model_path) and
        (os.path.getmtime(saved_model_path)
         > os.path.getmtime(res.twitter_logger_en_path))):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path,
                                                           binary=True,
                                                           unicode_errors='replace')
    else:
        logger.error('Custom3 model doesn\'t exist %s', saved_model_path)
        raise ValueError


def build_custom3(initial_model,
                  lexicon={},
                  a_i=0.5, b_ij=0.5, n_iter=10, in_place=True):
    """Retrofit a model using faruqui:2014:NIPS-DLRLW method.

    Args:
        in_place: Modify the given model instead of copying it if True."""
    if not in_place:
        initial_model_file = tempfile.NamedTemporaryFile(mode='w+',
                                                         encoding='utf-8',
                                                         delete=False)
        initial_model_file.close()
        initial_model.save(initial_model_file.name)
        model = gensim.models.Word2Vec.load(initial_model_file.name)
        os.remove(initial_model_file.name)
    else:
        model = initial_model
    old_lexicon = lexicon
    lexicon = {}
    for w in old_lexicon:
        if w in model:
            lexicon[w] = old_lexicon[w]
    lexicon_inv = utils.invert_dict_nonunique(lexicon)

    for it in range(n_iter):
        # loop through every node also in ontology (else just use data
        # estimate)
        for word in lexicon:
            if word not in model:
                continue
            i = model.vocab[word].index
            word_neighbours = [w for w in lexicon_inv[lexicon[word]]
                               if w != word]
            num_neighbours = len(word_neighbours)

            if b_ij == 'degree':
                b_ij = 1/num_neighbours

            #no neighbours, pass - use data estimate
            if num_neighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            model.syn0[i] = num_neighbours * a_i * initial_model.syn0[i]
            # loop over neighbours and add to new vector
            # for pp_word in word_neighbours:
            #     j = model.vocab[pp_word].index
            #     model.syn0[i] += b_ij * model.syn0[j]

            # Vectorized version of the above
            word_neighbours = [model.vocab[w].index for w in word_neighbours]
            model.syn0[i] = model.syn0[i] + b_ij * np.sum(model.syn0[word_neighbours], axis=0)
            model.syn0[i] = model.syn0[i] / (num_neighbours * (b_ij + a_i))
    return model


def build_custom3_1(initial_model,
                    lexicon={},
                    a_i=1, b_ij=1, c_ij=1, n_iter=10, in_place=True):
    """Derived from faruqui:2014:NIPS-DLRLW method.
Put same class closer and other classes away.

    Args:
        in_place: Modify the given model instead of copying it if True."""
    if not in_place:
        initial_model_file = tempfile.NamedTemporaryFile(mode='w+',
                                                         encoding='utf-8',
                                                         delete=False)
        initial_model_file.close()
        initial_model.save(initial_model_file.name)
        model = gensim.models.Word2Vec.load(initial_model_file.name)
        os.remove(initial_model_file.name)
    else:
        model = initial_model
    old_lexicon = lexicon
    lexicon = {}
    for w in old_lexicon:
        if w in model:
            lexicon[w] = old_lexicon[w]
    lexicon_inv = utils.invert_dict_nonunique(lexicon)

    for it in range(n_iter):
        # loop through every node also in ontology (else just use data
        # estimate)
        for word in lexicon:
            if word not in model:
                continue
            i = model.vocab[word].index
            word_neighbours = [w for w in lexicon_inv[lexicon[word]]
                               if w != word]

            # Non-neighbours are words with different classes than WORD
            word_not_neighbours = []
            for c in lexicon_inv:
                if c != lexicon[word]:
                    word_not_neighbours.extend(lexicon_inv[c])
            # Remove duplicate
            word_not_neighbours = list(set(word_not_neighbours))

            num_neighbours = len(word_neighbours)
            num_not_neighbours = len(word_not_neighbours)

            if b_ij == 'degree':
                b_ij = 1/num_neighbours

            if c_ij == 'degree':
                c_ij = 1/num_not_neighbours

            # FIXE use not_neighbours
            #no neighbours, pass - use data estimate
            if num_neighbours == 0 and num_not_neighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            model.syn0[i] = num_neighbours * a_i * initial_model.syn0[i]
            # loop over neighbours and add to new vector
            # for pp_word in word_neighbours:
            #     j = model.vocab[pp_word].index
            #     model.syn0[i] += b_ij * model.syn0[j]

            # Vectorized version of the above
            word_neighbours = [model.vocab[w].index for w in word_neighbours]
            model.syn0[i] = model.syn0[i] + b_ij * np.sum(model.syn0[word_neighbours], axis=0)
            word_not_neighbours = [model.vocab[w].index for w in word_not_neighbours]
            model.syn0[i] = model.syn0[i] - c_ij * np.sum(model.syn0[word_not_neighbours], axis=0)
            model.syn0[i] = model.syn0[i] / (num_neighbours * (b_ij + a_i))
    return model


def get_gnews():
    logger.info('Load gnews model')
    saved_model_path = res.gnews_negative300_path
    if os.path.exists(saved_model_path) and os.path.getmtime(saved_model_path) > os.path.getmtime(res.twitter_logger_en_path):
        return gensim.models.Word2Vec.load_word2vec_format(saved_model_path,
                                                           binary=True,
                                                           unicode_errors='replace')
    else:
        logger.error('Gnews model doesn\'t exist %s', saved_model_path)
        raise ValueError


def compare_model_with_lexicon_class(model, lexicon,
                                     **kwargs):
    lexicon_inv = utils.invert_dict_nonunique(lexicon)
    for c in lexicon_inv:
        c_lexicon = {}
        for w in lexicon_inv[c]:
            c_lexicon[w] = c
        logger.info('Compare with class %s', c)
        compare_model_with_lexicon(model, lexicon, **kwargs)


def compare_model_with_lexicon(model, lexicon,
                               topn=100,
                               sample_size=None,
                               clean_after=True,
                               normalize_word=True):
    """Compare model with lexicon with trec_eval script.

https://faculty.washington.edu/levow/courses/ling573_SPR2011/hw/trec_eval_desc.htm
http://trec.nist.gov/trec_eval/

./trec_eval qrel top

TOP reponse
| QID2  | ITER        | DOCNO | RANK        | SIM                    | RUN_ID |
|-------+-------------+-------+-------------+------------------------+--------|
|       | 0 (ignored) | word  | 1 (ignored) | similarity score float | RUN_ID |

QREL verite terrain
| QID | ITER        | DOCNO | REL |
|-----+-------------+-------+-----|
|     | 0 (ignored) |       |     |

QID = ID du mot

    Args:
        model: variable documentation.
        lexicon: variable documentation.
        topn: variable documentation.
        sample_size: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    logger.info('Build lexicon_index for qid (%s)', sample_size)
    if sample_size is None:
        sample_size = len(list(lexicon))
    else:
        sample_size = min(sample_size, len(list(lexicon)))

    if normalize_word:
        model_vocab = [w_norm(w) for w in model.vocab]
    else:
        model_vocab = list(model.vocab)

    lexicon_index = list(enumerate([word for word
                                    in random.sample(list(lexicon),
                                                     sample_size)
                                    if word in model_vocab]))

    lexicon_inv = utils.invert_dict_nonunique(lexicon)

    qrel_file = tempfile.NamedTemporaryFile(mode='w+',
                                            encoding='utf-8',
                                            delete=False,
                                            prefix='qrel')
    logger.info('Build Ground Truth Qrel file (%s)', qrel_file.name)

    for qid, word in lexicon_index:
        for docno in lexicon_inv[lexicon[word]]:
            qrel_file.write('%d 0 %s 1\n' % (qid, docno))

    top_file = tempfile.NamedTemporaryFile(mode='w+',
                                           encoding='utf-8',
                                           delete=False,
                                           prefix='top')
    logger.info('Build Top (%d) answer from the model (%s)',
                topn, top_file.name)

    for qid, word in lexicon_index:
        seen_docno = {}
        word_in_vocab = list(model.vocab)[model_vocab.index(word)]
        for (rank, (docno, sim)) in enumerate(model.most_similar(word_in_vocab,
                                                                 topn=topn)):
            if docno == '' or docno in seen_docno or not re.match(r'^[a-z]+$', docno):
                continue
            seen_docno[docno] = 1
            top_file.write('%d 0 %s %d %f runid\n' % (qid, docno, rank, sim))
            if len(seen_docno) == topn:
                break

    logger.info('Run trec_eval script')
    ret = None
    try:
        p = Popen(['./trec_eval',
                   '-m', 'all_trec',
                   '-m', 'P.1,2,5,10,25,50,100,200,500,1000',
                   qrel_file.name,
                   top_file.name],
                  stdout=PIPE, stderr=PIPE, cwd=res.TREC_EVAL_PATH)
        out, err = p.communicate()
        ret = out + err
        ret = ret.decode()
        logger.info(ret)
    except Exception:
        pass
    if clean_after:
        os.remove(qrel_file.name)
        os.remove(top_file.name)

    return ret


def old_compare_model_with_lexicon(model, lexicon,
                               topn=100,
                               sample_size=None,
                               clean_after=True,
                               normalize_word=True):
    """Compare model with lexicon with trec_eval script.

https://faculty.washington.edu/levow/courses/ling573_SPR2011/hw/trec_eval_desc.htm
http://trec.nist.gov/trec_eval/

./trec_eval qrel top

TOP reponse
| QID2  | ITER        | DOCNO | RANK        | SIM                    | RUN_ID |
|-------+-------------+-------+-------------+------------------------+--------|
|       | 0 (ignored) | word  | 1 (ignored) | similarity score float | RUN_ID |

QREL verite terrain
| QID | ITER        | DOCNO | REL |
|-----+-------------+-------+-----|
|     | 0 (ignored) |       |     |

QID = ID du mot

    Args:
        model: variable documentation.
        lexicon: variable documentation.
        topn: variable documentation.
        sample_size: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    logger.info('Build lexicon_index for qid (%s)', sample_size)
    if sample_size is None:
        sample_size = len(list(lexicon))
    else:
        sample_size = min(sample_size, len(list(lexicon)))

    if normalize_word:
        model_vocab = [w_norm(w) for w in model.vocab]
    else:
        model_vocab = list(model.vocab)

    lexicon_index = list(enumerate([word for word
                                    in random.sample(list(lexicon),
                                                     sample_size)
                                    if word in model_vocab]))

    lexicon_inv = utils.invert_dict_nonunique(lexicon)

    qrel_file = tempfile.NamedTemporaryFile(mode='w+',
                                            encoding='utf-8',
                                            delete=False,
                                            prefix='qrel')
    logger.info('Build Ground Truth Qrel file (%s)', qrel_file.name)

    for qid, word in lexicon_index:
        for docno in lexicon_inv[lexicon[word]]:
            qrel_file.write('%d 0 %s 1\n' % (qid, docno))

    top_file = tempfile.NamedTemporaryFile(mode='w+',
                                           encoding='utf-8',
                                           delete=False,
                                           prefix='top')
    logger.info('Build Top (%d) answer from the model (%s)',
                topn, top_file.name)

    for qid, word in lexicon_index:
        seen_docno = {}
        word_in_vocab = list(model.vocab)[model_vocab.index(word)]
        for (rank, (docno, sim)) in enumerate(model.most_similar(word_in_vocab,
                                                                 topn=topn)):
            if docno == '' or docno in seen_docno or not re.match(r'^[a-z]+$', docno):
                continue
            seen_docno[docno] = 1
            top_file.write('%d 0 %s %d %f runid\n' % (qid, docno, rank, sim))
            if len(seen_docno) == topn:
                break

    logger.info('Run trec_eval script')
    ret = None
    try:
        p = Popen(['./trec_eval',
                   '-m', 'all_trec',
                   '-m', 'P.1,2,5,10,25,50,100,200,500,1000',
                   qrel_file.name,
                   top_file.name],
                  stdout=PIPE, stderr=PIPE, cwd=res.TREC_EVAL_PATH)
        out, err = p.communicate()
        ret = out + err
        ret = ret.decode()
        logger.info(ret)
    except Exception:
        pass
    if clean_after:
        os.remove(qrel_file.name)
        os.remove(top_file.name)

    return ret

