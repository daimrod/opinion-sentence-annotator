#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import tempfile
from subprocess import PIPE, Popen
import os
import random
import codecs
import math

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from resources import SEMEVAL_SCORER_PATH

logger = logging.getLogger(__name__)


def pretty_pipeline(obj):
    """Pretty print a sklearn Pipeline.

    This function is especially useful to extract information within
    FeatureUnion Pipeline.

    Args:
        obj: A sklearn Pipeline.

    Returns:
        A flat version of the Pipeline object.

    """
    if isinstance(obj, list):
        return [pretty_pipeline(o) for o in obj]
    elif isinstance(obj, FeatureUnion):
        return {'n_jobs': obj.n_jobs,
                'transformer_list': obj.transformer_list,
                'transformer_weights': obj.transformer_weights}
    elif isinstance(obj, Pipeline):
        return {'steps': pretty_pipeline(obj.steps)}
    elif isinstance(obj, tuple):
        return pretty_pipeline(list(obj))
    else:
        return obj


def strings_to_integers(strings, labels):
    """Convert an array of strings to an array of integers.

    Convert an array of strings to an array of integers where the same
    string will always have the same integers value.

    Args:
        strings: An array of strings.

    Returns:
        An array of integers
    """
    integers = []
    for string in strings:
        integers.append(labels.index(string))
    return integers


def integers_to_strings(integers, labels):
    """Convert an array of integers to an array of strings using labels as
reference.

    Args:
        integers: An array of integers.
        labels: An array of strings where each integers will be
        replaced by the string at the index.

    Returns:
        An array of strings.
    """
    strings = []
    for integer in integers:
        strings.append(labels[integer])
    return strings


def merge_classes(lst, classes, new_class):

    """Merge classes from lst into one new_class.

    Args:
        lst: A list of classes that will be replaced (strings).
        classes: A list of classes to replace (strings).
        new_class: The new class (string).

    Returns:
        The list with all occurences of classes replaced by new_class.
    """
    for i in range(len(lst)):
        if lst[i] in classes:
            lst[i] = new_class
    return lst


def eval_with_semeval_script(test, predicted):
    """Eval prediction on test with semeval script (T4SA).

    Args:
        test: variable documentation.
        predicted: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    predicted = integers_to_strings(predicted, test.labels)
    ofile = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    ret = None
    try:
        for (sid, pred) in zip(test.sid, predicted):
            ofile.write('%s\t%s\n' % (sid, pred))
        ofile.close()
        p = Popen(['_scripts/SemEval2016_task4_test_scorer_subtaskA.pl',
                   ofile.name], stdout=PIPE, stderr=PIPE, cwd=SEMEVAL_SCORER_PATH)
        out, err = p.communicate()
        ret = out + err
        ret = ret.decode()
        with open(ofile.name + '.scored', 'r') as ifile:
            for line in ifile:
                ret += line
    finally:
        if not ofile.closed:
            ofile.close()
        os.remove(ofile.name)
        os.remove(ofile.name + '.scored')
    return ret


def assoc_value(lst, value):
    """Return the element associated to value and its index.

    Args:
        lst: A associative array/list.
        value: The value to match on.

    Returns:
        The element associated to value (with the value itself) and
        its index.
    """
    for (idx, el) in enumerate(lst):
        if el[0] == value:
            return el, idx


def invert_dict_nonunique(d):
    newdict = {}
    for k in d:
        newdict.setdefault(d[k], []).append(k)
    return newdict


def split_train_valid(input_path, valid_num=3000):
    """Split a file in two (.valid and .train) with valid_num lines in
.valid and everything else in .train.
    """
    train_path = input_path + '.train'
    valid_path = input_path + '.valid'
    nb_line = 0
    with codecs.open(input_path, 'r', 'utf-8') as ifile:
        nb_line = len([line for line in ifile])
    if valid_num <= 1 and valid_num >= 0:
        valid_num = math.floor(nb_line * valid_num)

    valid_indexes = random.sample(range(nb_line), valid_num)
    try:
        ifile = codecs.open(input_path, 'r', 'utf-8')
        train_file = codecs.open(train_path, 'w+', 'utf-8')
        valid_file = codecs.open(valid_path, 'w+', 'utf-8')
        idx = 0
        for line in ifile:
            try:
                v_idx = valid_indexes.index(idx)
                valid_file.write(line)
                del valid_indexes[v_idx]
            except ValueError:
                train_file.write(line)
            idx += 1
    finally:
        ifile.close()
        train_file.close()
        valid_file.close()


def opinion_lexicon_to_graph(lexicon):
    """Return a undirected graph from lexicon.

LEXICON is an opinion lexicon where each key is a class and the value
associated to it is a list of words that belongs to the class.

This function will build a undirected graph where each node are words
and the edges between nodes represent a similarity relationship. There
will be an edge between two words if they belong to the same class.

In practice, this method returns a dictionnary where a key is a word
of LEXICON and the value associated to it are all words from the same
class.

This is intended to be used by emb.build_custom3
"""
    ret = {}
    lexicon_inv = invert_dict_nonunique(lexicon)
    for c in lexicon_inv:
        words = lexicon_inv[c]
        for word in words:
            ret[word] = words
    return ret


def split_lexicon_train_test(lexicon, ratio=0.9, shuffle=False):
    """Split each class of the lexicon in train and test.

    Args:
        lexicon: A lexicon to split.
        ratio: The ratio of train/test. 0.9 means 90% of the lexicon
        will go in the train lexicon.
        shuffle: A boolean to specify that the lexicon should be
        shuffled before splitting.

    Returns:
        A train lexicon and a test lexicon.
    """
    train_lexicon = {}
    test_lexicon = {}
    lexicon_inv = invert_dict_nonunique(lexicon)
    for c in lexicon_inv:
        c_words = lexicon_inv[c]
        n = len(c_words)
        if shuffle:
            random.shuffle(c_words)
        limit = math.floor(n * ratio)
        for w in c_words[:limit]:
            train_lexicon[w] = c
        for w in c_words[limit:]:
            test_lexicon[w] = c

    return train_lexicon, test_lexicon


def remove_multi_words_in_lexicon(lexicon):
    """Remove multi-words in lexicon"""
    ret = {}
    for w in lexicon:
        if len(w.split(' ')) == 1:
            ret[w] = lexicon[w]
    return ret
