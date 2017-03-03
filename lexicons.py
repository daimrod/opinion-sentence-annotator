#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


logger = logging.getLogger(__name__)

import reader
import resources as res

# A global variables used to store known lexicons
_lexicons = {}


def get_lexicon(lexicon_name):
    """Return the lexicon designated by lexicon_name.

    Args:
        lexicon_name: The name of a lexicon.

    Returns:
        Returns the requested lexicon
    """
    if lexicon_name not in _lexicons:
        raise KeyError('The lexicon \'%s\' has not been registered' % lexicon_name)
    lexicon_reader, lexicon_path = _lexicons[lexicon_name]
    return lexicon_reader(lexicon_path)


def register_lexicon(lexicon_name, lexicon_reader, lexicon_path):
    """Register a lexicon and how to read it.

    This function register into a datastructure how to load a lexicon.

    Args:
        lexicon_name: The name of a lexicon.
        lexicon_reader: A function to read the given lexicon.
        lexicon_path: The path to read the given lexicon.

    Returns:
        Nothing"""
    _lexicons[lexicon_name] = (lexicon_reader, lexicon_path)


register_lexicon('bing_liu', reader.read_bing_liu, res.bing_liu_lexicon_path)
register_lexicon('mpqa', reader.read_mpqa, res.mpqa_lexicon_path)
register_lexicon('mpqa_plus', reader.read_mpqa_plus, res.mpqa_plus_lexicon_path)
register_lexicon('nrc_emotion', reader.read_nrc_emotion, res.nrc_emotion_lexicon_path)
register_lexicon('nrc_emotions', reader.read_nrc_emotions, res.nrc_emotion_lexicon_path)
register_lexicon('nrc_hashtag_unigram', reader.read_nrc_hashtag_unigram, res.nrc_hashtag_unigram_lexicon_path)
register_lexicon('nrc_hashtag_bigram', reader.read_nrc_hashtag_bigram, res.nrc_hashtag_bigram_lexicon_path)
register_lexicon('nrc_hashtag_pair', reader.read_nrc_hashtag_pair, res.nrc_hashtag_pair_lexicon_path)
register_lexicon('nrc_hashtag_sentimenthashtags', reader.read_nrc_hashtag_sentimenthashtags, res.nrc_hashtag_sentimenthashtags_lexicon_path)
register_lexicon('lidilem_adjectifs', reader.read_lidilem_adjectifs, res.lidilem_adjectifs_lexicon_path)
register_lexicon('lidilem_noms', reader.read_lidilem_noms, res.lidilem_noms_lexicon_path)
register_lexicon('lidilem_verbes', reader.read_lidilem_verbes, res.lidilem_verbes_lexicon_path)
register_lexicon('blogoscopie', reader.read_blogoscopie, res.blogoscopie_lexicon_path)
