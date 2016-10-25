#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline


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


def strings_to_integers(strings):
    """Convert an array of strings to an array of integers.

    Convert an array of strings to an array of integers where the same
    string will always have the same integers value.

    Args:
        strings: An array of strings.

    Returns:
        An array of integers
    """
    labels = list(set(strings))
    integers = []
    for string in strings:
        integers.append(labels.index(string))
    return integers, labels


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
