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
    logger.addHandler(sh)

    # FileHandler
    fh = logging.FileHandler('log.txt', 'a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

from reader import TwitterLoggerTextReader
from reader import URLReplacer
from reader import UserNameReplacer
from reader import Tokenizer
from reader import Splitter
from reader import FuzzyOpinionRecognizer
import resources as res
import features as feat
import codecs


def main():
    """Select tweets with emotions.

    Args:
        ipath: variable documentation.
        opath: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    reader = TwitterLoggerTextReader(res.twitter_logger_en_path)
    reader = URLReplacer(reader)
    reader = UserNameReplacer(reader)
    reader = Tokenizer(reader, feat.happyfuntokenizer)
    reader = Splitter(reader)

    happy_set = set()
    sad_set = set()
    with codecs.open(res.happy_emoticons_path, 'r', 'utf-8') as ifile:
        for line in ifile:
            emoticon = line.strip()
            happy_set.add(emoticon)
    with codecs.open(res.sad_emoticons_path, 'r', 'utf-8') as ifile:
        for line in ifile:
            emoticon = line.strip()
            sad_set.add(emoticon)
    reader = FuzzyOpinionRecognizer(reader, happy_set, sad_set)
    n = 0
    for s in reader:
        n += 1
        if n > 10:
            break
        print(s)
    return reader
