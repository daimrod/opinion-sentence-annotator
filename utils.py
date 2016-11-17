#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import tempfile
from subprocess import PIPE, Popen
import os
from resources import SEMEVAL_SCORER_PATH


if 'logger' not in locals():
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
