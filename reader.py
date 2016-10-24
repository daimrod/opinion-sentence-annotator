#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import codecs
from lxml import etree


if 'logger' not in locals():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
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


########## Data Reader
class Dataset():
    def __init__(self,
                 data=[],
                 filenames=[],
                 target_names=[],
                 target=[],
                 uid=[], sid=[]):
        self.data = data
        self.filenames = filenames
        self.target_names = target_names
        self.target = target
        self.uid = uid
        self.sid = sid

    def truncate(self, n):
        self.data = self.data[:n]
        self.filenames = self.filenames[:n]
        self.target_names = self.target_names[:n]
        self.target = self.target[:n]
        self.uid = self.uid[:n]
        self.sid = self.sid[:n]


def read_semeval_dataset(ipath, separator='\t',
                         ignore_not_available=True):
    """Return a dataset following sklearn format.

       Convert the data from SEMEVAL Sentiment Analysis Task of
       Message Polarity Detection to sklearn format.

       The sklearn format is presented here :
       http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

       train.data[N] = the data themselves, e.g. the strings
       train.filenames[N] = the filename from wich the data come from
       train.target_names[N] = the name of the class associated with each entry
       train.target[N] = the class of each entry converted to integer

       The format of the SEMEVAL dataset for the Message Polarity task
       can be the following :
       <SID><tab><UID><tab><TOPIC><tab><positive|negative|neutral|objective><tab><TWITTER_MESSAGE>
       <UID><tab><TOPIC><tab><positive|negative|neutral|objective><tab><TWITTER_MESSAGE>
       Where the polarity can be between double-quotes or not and TEXT
       is the content of the tweet or 'Not Available' if the tweet
       doesn't exist anymore.

    Args:
        ipath: The path to the dataset in SEMEVAL format.
        separator: The separator between attributes in a line.
        ignore_not_available: Set to True to ignore tweets that aren't
        available in the SEMEVAL corpus (we weren't able to download
        them).
    Returns:
        A Dataset with filled with the data from ipath.

    Raises:
        IOError: An error occurred.
    """
    dataset = Dataset(data=[],
                      filenames=[],
                      target_names=[],
                      target=[],
                      sid=[],
                      uid=[])
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        for line in ifile:
            line = line.strip()
            if line:
                try:
                    [sid, uid, polarity, text] = line.split(separator,
                                                            maxsplit=4)
                    if ignore_not_available and text == 'Not Available':
                        continue
                    dataset.sid.append(sid)
                    dataset.uid.append(uid)
                    dataset.target_names.append(polarity.replace('"', ''))
                    dataset.data.append({'text': text})
                except ValueError:
                    try:
                        [uid, polarity, text] = line.split(separator,
                                                           maxsplit=3)
                        if ignore_not_available and text == 'Not Available':
                            continue
                        dataset.uid.append(uid)
                        dataset.target_names.append(polarity.replace('"', ''))
                        dataset.data.append({'text': text})
                    except ValueError:
                        logging.warn('Couldn\'t parse line %s', line)
    return dataset


def read_bing_liu(neg_path, pos_path):
    """Return a dictionary of negative/positive words.

    Args:
        neg_path: variable documentation.
        pos_path: variable documentation.

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    for (path, c) in [(neg_path, 'negative'),
                      (pos_path, 'positive')]:
        with codecs.open(path, 'r', 'utf-8') as ifile:
            for word in ifile:
                word = word.strip()
                if word and not word.startswith(';'):
                    ret[word] = c
    return ret


def read_mpqa(mpqa_path):
    """Return a dictionary of negative/positive words.

    Args:
        neg_path: variable documentation.
        pos_path: variable documentation.

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(mpqa_path, 'r', 'utf-8') as ifile:
        for line in ifile:
            line = line.strip()
            cols = line.split()
            if len(cols) == 6:
                word = '='.join(cols[2].split('=')[1:])
                polarity = '='.join(cols[5].split('=')[1:])
                ret[word] = polarity
    return ret


def read_mpqa_plus(mpqa_path_plus):
    """Return a dictionary of negative/positive words.

    Args:

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(mpqa_path_plus, 'r', 'utf-8') as ifile:
        tree = etree.parse(ifile)
    root = tree.getroot()
    for lexical_entry in root.iterchildren():
        word = None
        polarity = None
        for el in lexical_entry.iterchildren():
            if el.tag == 'morpho':
                for node in el.iterchildren():
                    if node.tag == 'name':
                        word = node.text
            if el.tag == 'evaluation':
                polarity = el.get('subtype')
        if word is not None and polarity is not None:
            ret[word] = polarity
    return ret


def read_nrc_hashtag(path):
    """Return a dictionary of words with their scores.

    Args:
        path: Path to NRC Hashtag lexicon.

    Returns:
        Returns a dictionary of words with their scores.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(path, 'r', 'utf-8') as ifile:
        for line in ifile:
            line = line.strip()
            elements = line.split('\t')
            if len(elements) != 4:
                logger.error('Couldn\'t parse line')
                logger.error(line)
            else:
                term, score, _, _ = elements
                ret[term] = float(score)
    return ret


def read_nrc_emotion(nrc_path):
    """Return a dictionary of negative/positive words.

    Args:
        nrc_path: variable documentation.

    Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(nrc_path, 'r', 'utf-8') as ifile:
        for line in ifile:
            line = line.strip()
            word, affect, valence = line.split('\t')
            if affect == 'positive' or affect == 'negative':
                ret[word] = affect
    return ret


def read_carnegie_clusters(path):
    """Return a lexicon representation of carnegie clusters.

    Args:
        path: Path to Carnegie clusters.

    Returns:
        A dictionnary of words in Carnegie clusters.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with open(path, 'rb') as ifile:
        last_line = '#EMPTY#'
        for line in ifile:
            line = line.strip().decode('utf-8')
            elements = line.split('\t')
            if len(elements) != 3:
                logger.error("Couldn't read Carnegie clusters properly")
                logger.error(last_line)
                logger.error(elements)
            else:
                cluster_id, word, count = elements
                ret[word] = cluster_id
            last_line = line
    return ret
