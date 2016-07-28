#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import happyfuntokenizing
from nltk.tokenize import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def happyfuntokenizer(s):
    """Tokenize a string with happyfuntokenizing.py.

    Args:
        s: A string to tokenize.

    Returns:
        The string tokenized.
    """
    tok = happyfuntokenizing.Tokenizer()
    return ' '.join(tok.tokenize(s))


def nltktokenizer(s):
    """Tokenize a string with NLTK Twitter Tokenizer.

    Args:
        s: A string to tokenize.

    Returns:
        The string tokenized.
    """
    tok = TweetTokenizer()
    return ' '.join(tok.tokenize(s))


class Tokenizer(BaseEstimator, TransformerMixin):
    """Tokenize.

    """
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tok = lambda s: s
        else:
            self.tok = tokenizer

    def fit(self, x, y=None):
        return self

    def transform(self, strings):
        return [self.tok(s) for s in strings]
