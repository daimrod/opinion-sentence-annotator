#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from sklearn.base import BaseEstimator, TransformerMixin

import functools
import itertools
import re
import numbers
import numpy as np
import scipy
import codecs
import tempfile
import random

from subprocess import Popen, PIPE

from resources import happy_emoticons_path
from resources import sad_emoticons_path
from resources import SENNA_PATH

import happyfuntokenizing
from nltk.tokenize import TweetTokenizer
import nltk

import utils

if 'logger' not in locals():
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


########## Features Extractions
##### Utils
class ItemExtractor(BaseEstimator, TransformerMixin):
    """Extract a particular entry in the input dictionnary.

    Attributes:
        item: The item to extract.
    """
    def __init__(self, item):
        self.item = item

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        return [x[self.item] for x in X]


class ApplyFunction(BaseEstimator, TransformerMixin):
    """Apply a function to each entry.

    Attributes:
        fun: The function to apply.
    """
    def __init__(self, fun):
        self.fun = fun

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        return [self.fun(x) for x in X]


class MeanVectors(BaseEstimator, TransformerMixin):
    """Concatenate the input vectors using the mean.

    Attributes:
        Nothing
    """
    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        return [np.mean(x, axis=0) for x in X]


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """Extract main features.

    Attributes:
        features: A dictionnary with features to extract of the form
        {feature_name: extractor} where extractor is a function.
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        ret = []
        for x in X:
            d = {}
            for (f_name, f) in self.features:
                extracted = f(x)
                if type(extracted) is dict:
                    d.update(extracted)
                else:
                    d[f_name] = extracted
            ret.append(d)
        return ret


class FindClosestInLexicon(object):
    """
    """
    def __init__(self, model, lexicon, topn, lower=True):
        self.model = model
        self.lexicon = lexicon
        self.topn = topn
        self.lower = lower

        self.lexicon_inv = {}
        for word_lex in lexicon:
            if word_lex not in model:
                continue
            polarity = lexicon[word_lex]
            if polarity not in self.lexicon_inv:
                self.lexicon_inv[polarity] = []
            self.lexicon_inv[polarity].append(model[word_lex])

    # def find_closest_in_lexicon(self, word):
    #     if word not in self.model:
    #         return [-1] * len(self.lexicon_inv)

    #     ret = []
    #     for polarity in self.lexicon_inv:
    #         if word in self.lexicon:
    #             if polarity == self.lexicon[word]:
    #                 score = 1
    #             else:
    #                 score = -1
    #         else:
    #             polarized = self.lexicon_inv[polarity]
    #             cdist = scipy.spatial.distance.cdist([self.model[word]],
    #                                                  polarized, metric='cosine')
    #             cdist = (cdist - 1) * -1
    #             score = np.sort(cdist)[0][-1]
    #         ret.append(score)
    #     return ret

    @functools.lru_cache(maxsize=None)
    def find_scores_closest_in_lexicon(self, word):
        """Return an array of scores [score1, score2, ...].

        Return an array of scores [score1, score2, ...] with one score
for each class in the given lexicon. The score is the distance to the
closest word of the given class.
        """
        if self.lower:
            word = word.lower()

        if word not in self.model:
            return [-1] * len(self.lexicon_inv)

        ret = []
        for polarity in self.lexicon_inv:
            polarized = self.lexicon_inv[polarity]
            cdist = scipy.spatial.distance.cdist([self.model[word]],
                                                 polarized, metric='cosine')
            cdist = (cdist - 1) * -1
            score = np.sort(cdist)[0][-1]
            ret.append(score)
        return ret

    @functools.lru_cache(maxsize=None)
    def find_rank_closest_in_lexicon(self, word):
        """Return an array of ranks [rank1, rank2, ...].

        Return an array of scores [rank1, rank2, ...] with the rank
        for each class in the given lexicon. The rank is the rank of
        the closest word of the given class.

        If we can't find a closest_word to word for a class, then its rank is set to topn*2/topn.
        """
        if self.lower:
            word = word.lower()
        ret = [self.topn*2/self.topn] * len(self.lexicon_inv)
        if word not in self.model:
            return ret

        closest_words = self.model.similar_by_word(word, topn=self.topn)
        classes = list(self.lexicon_inv)
        done_classes = []
        for (rank, el) in enumerate(closest_words):
            close_word, score = el
            if close_word in self.lexicon:
                word_class = self.lexicon[close_word]
                if word_class not in done_classes:
                    class_idx = classes.index(word_class)
                    offset = class_idx
                    ret[offset] = rank / self.topn
        return ret

    # @functools.lru_cache(maxsize=None)
    # def find_closest_in_lexicon(self, word, debug=False):
    #     """Find the closest words for the given word in the model for each
    #     polarity in the lexicon.

    #     Longer function information

    #     Args:
    #         word: A word.
    #         model: A gensim model.
    #         lexicon: A lexicon.

    #     Returns:
    #         Returns an array with the closest words for each polarity.

    #     """
    #     lexicon_inv = {}
    #     lexicon_inv2 = {}
    #     for word_lex in self.lexicon:
    #         polarity = self.lexicon[word_lex]
    #         if polarity not in lexicon_inv:
    #             lexicon_inv[polarity] = -1
    #             lexicon_inv2[polarity] = None
    #         if word not in self.model or word_lex not in self.model:
    #             dist = -1
    #         else:
    #             dist = self.model.similarity(word, word_lex)
    #         if lexicon_inv[polarity] < dist:
    #             lexicon_inv[polarity] = dist
    #             lexicon_inv2[polarity] = word_lex
    #     # if debug:
    #     #     return lexicon_inv, lexicon_inv2
    #     # else:
    #     #     return [lexicon_inv[pol] for pol in lexicon_inv]
    #     return [lexicon_inv[pol] for pol in lexicon_inv]


## Features generators
class F_Emoticons(object):
    def __init__(self):
        happy_set = set()
        sad_set = set()
        with codecs.open(happy_emoticons_path, 'r', 'utf-8') as ifile:
            for line in ifile:
                emoticon = line.strip()
                happy_set.add(emoticon)
        with codecs.open(sad_emoticons_path, 'r', 'utf-8') as ifile:
            for line in ifile:
                emoticon = line.strip()

        self.happy_set = happy_set
        self.sad_set = sad_set

    def f_emoticons(self, s):
        """Return informations about emoticons.

        - presence/absence of positive and negative emoticons at any
          position in the tweet;
        - whether the last token is a positive or negative emoticon;

        Args:
            s: variable documentation.

        Returns:
            A vector representation of emoticons information in s.
        """
        has_happy = False
        has_sad = False
        for word in s.split(' '):
            if word in self.happy_set:
                has_happy = True
            elif word in self.sad_set:
                has_sad = True

        if s[-1] in self.happy_set:
            last_tok = 1
        elif s[-1] in self.sad_set:
            last_tok = -1
        else:
            last_tok = 0
        return [has_happy * 1, has_sad * 1, last_tok]
    __call__ = f_emoticons


def f_n_neg_context(s):
    """Return the number of negated contexts.

    Args:
        s: A string.

    Returns:
        The number of negated contexts.

    """
    re_beg_ctxt = r"(\b(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b|n't\b)"
    re_end_ctxt = r"[,.:;!?]+"
    in_neg_ctxt = False
    neg_ctxt_counted = False
    n_neg_context = 0
    for word in s.split(' '):
        if re.search(re_end_ctxt, word, flags=re.IGNORECASE) and in_neg_ctxt:
            in_neg_ctxt = False
            neg_ctxt_counted = False
        elif re.search(re_beg_ctxt, word, flags=re.IGNORECASE):
            in_neg_ctxt = True
            if not neg_ctxt_counted:
                n_neg_context += 1
            neg_ctxt_counted = True
    return [n_neg_context]


def f_all_caps(s):
    """Return the number of words with all characters in upper case.

    This function assumes that the string is tokenized and all words
    are separated by spaces.

    Args:
        s: A string.

    Returns:
        The number of words with all characters in upper case.
    """
    n = 0
    for word in s.split(' '):
        if word.isupper():
            n = n + 1
    return [n]


def f_n_hashtags(s):
    """Return the number of hashtags in the string.

    Args:
        s: A string.

    Returns:
        The number of hashtags.
    """
    n = 0
    for word in s.split(' '):
        if word.startswith('#'):
            n += 1
    return [n]


def f_punctuation(s):
    """Return information about the punctuation in the input string.

    - the number of contiguous sequences of exclamation marks,
      question marks, and both exclamation and question marks;
    - whether the last token contains exclamation or question mark;

    Args:
        s: variable documentation.

    Returns:
        - the number of contiguous sequences of exclamation marks,
          question marks, and both exclamation and question marks;
        - whether the last token contains exclamation or question mark;
    """
    continuous_excl = 0
    continuous_quest = 0
    continuous_excl_quest = 0

    excl_flag = 0
    quest_flag = 0
    excl_quest_flag = 0
    for char in s:
        if char == '!':
            excl_flag += 1
        else:
            if excl_flag > 1:
                continuous_excl += 1
            excl_flag = 0
        if char == '?':
            if quest_flag > 1:
                continuous_quest += 1
            quest_flag += 1
        else:
            quest_flag = 0
        if char == '!' or char == '?':
            excl_quest_flag += 1
        else:
            excl_quest_flag = 0
            if excl_quest_flag > 1:
                continuous_excl_quest += 1
    last_word = s.split(' ')[-1]
    last_excl_or_quest = '!' in last_word or '?' in last_word

    return [continuous_excl,
            continuous_quest,
            continuous_excl_quest,
            last_excl_or_quest * 1]


def f_elongated_words(s):
    """Return the number of words with one character repeated more than 2
times.

    This function assumes that the string is tokenized and all words
    are separated by spaces.

    Args:
        s: A string.

    Returns:
        The number of words with one character repeated more than 2 times.
    """
    n = 0
    for word in s.split(' '):
        for i in range(len(word) - 2):
            s = word[i:3+i]
            if len(set(s)) == 1:
                n = n + 1
                break
    return [n]


def f_all_syntax(s):
    ret = []
    for f in [f_elongated_words,
              f_all_caps]:
        ret.extend(f(s))
    return ret


##### Senna
def f_senna(s):
    """Return the string parsed with senna.

    Args:
        s: A string to parse.

    Returns:
        A dictionnary with POS, CHK and NER annotations for each word.

    Raises:
        IOError: An error occurred.
    """
    ret = {'POS': [], 'CHK': [], 'NER': []}
    p = Popen(['./senna', '-notokentags',
               '-usrtokens',
               '-pos', '-chk', '-ner'],
              stdin=PIPE,
              stdout=PIPE,
              stderr=PIPE, cwd=SENNA_PATH)
    out, err = p.communicate(input=s.encode())
    if p.returncode != 0:
        print(err.replace('*', '#'))
    out = out.decode()

    for line in out.split('\n'):
        line = line.strip()
        if not line:
            continue
        tags = line.split('\t')
        tags = [x.strip() for x in tags]
        pos, chk, ner = tags
        ret['POS'].append(pos)
        ret['CHK'].append(chk)
        ret['NER'].append(ner)
    return ret


def f_senna_multi(lst):
    """Return the list of strings parsed with senna.

    Args:
        lst: A list of strings to parse.

    Returns:
        A list of dictionnaries with POS, CHK and NER annotations for
        each word.

    Raises:
        IOError: An error occurred.
    """
    ret = []
    with tempfile.TemporaryFile() as temp_file:
        for s in lst:
            temp_file.write(s.encode())
            temp_file.write(b'\n')
        temp_file.seek(0, 0)
        p = Popen(['./senna', '-notokentags',
                   '-usrtokens',
                   '-pos', '-chk', '-ner'],
                  stdin=temp_file,
                  stdout=PIPE,
                  stderr=PIPE, cwd=SENNA_PATH)
    out, err = p.communicate(input=s.encode())
    if p.returncode != 0:
        print(err.replace('*', '#'))
    out = out.decode()

    d = {'pos': [], 'chk': [], 'ner': []}
    for line in out.split('\n'):
        line = line.strip()
        if not line:
            if d['pos'] != []:
                d['pos'] = ' '.join(d['pos'])
                d['chk'] = ' '.join(d['chk'])
                d['ner'] = ' '.join(d['ner'])
                ret.append(d)
                d = {'pos': [], 'chk': [], 'ner': []}
            continue
        tags = line.split('\t')
        tags = [x.strip() for x in tags]
        pos, chk, ner = tags
        d['pos'].append(pos)
        d['chk'].append(chk)
        d['ner'].append(ner)
    return ret


class F_Gensim_Model(object):
    def __init__(self, model):
        self.model = model
    """Make a feature generator with the given gensim model.

    Replace each words by its vector.

    Args:
        model: A gensim model.

    Returns:
        A feature generator for the given model.
    """
    def f_gensim_model(self, s):
        ret = []
        for word in s.split(' '):
            word = word.lower()
            if word in self.model:
                rep = self.model[word]
            else:
                rep = [0] * 100
            ret.extend(rep)
        return ret
    __call__ = f_gensim_model


class F_Find_Closest_In_Lexicon(object):
    def __init__(self, model, lexicon, topn):
        self.model = model
        self.lexicon = lexicon
        self.finder = FindClosestInLexicon(model, lexicon, topn)
        self.topn = topn

    def f_find_closest_in_lexicon(self, s):
        ret = []
        for word in s.split(' '):
            ret.append(self.finder.find_rank_closest_in_lexicon(word))
        return ret
    __call__ = f_find_closest_in_lexicon


class Array_To_Feature(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def array_to_feature(self, array_of_arrays):
        ret = {}
        for (idx1, array) in enumerate(array_of_arrays):
            for (idx2, val) in enumerate(array):
                ret['%s-%d-%d' % (self.prefix, idx1, idx2)] = val
        return ret
    __call__ = array_to_feature


class F_NRC_Project_Lexicon(object):
    """Make a feature generator with the given lexicon like NRC.

    For each lexicon and each polarity we calculated:
      - total count of tokens in the tweet with score greater than 0;
      - the sum of the scores for all tokens in the tweet;
      - the maximal score;
      - the non-zero score of the last token in the tweet;
    The lexicon features were created for all tokens in the tweet, for
    each part-of-speech tag, for hashtags, and for all-caps tokens.

    Args:
        lexicon: A lexicon.

    Returns:
        A feature generator for the given lexicon.
    """
    def __init__(self, lexicon, ngrams=1, use_pair=False):
        self.lexicon = lexicon
        self.ngrams = ngrams
        if use_pair:
            self.get_items = self.get_pair_items

    def get_items(self, s):
        return [' '.join(item) for item in nltk.ngrams(s.split(' '), self.ngrams)]

    def get_pair_items(self, s):
        s = s.split(' ')
        ngrams = itertools.chain(nltk.ngrams(s, 2), nltk.ngrams(s, 1))
        for i1, i2 in itertools.combinations(ngrams, 2):
            yield ' '.join(i1) + '---' + ' '.join(i2)

    def word_to_score(self, word):
        has_neg = '_NEG' in word
        word = re.sub('_NEG', '', word)
        if word in self.lexicon:
            score = self.lexicon[word]
            if score == 'positive':
                score = 1
            elif score == 'negative':
                score = -1
            # MPQA has the neutral and both annotations
            elif score == 'neutral' or score == 'both':
                score = 0
            elif not isinstance(score, numbers.Real):
                logger.error('Cannot determine what to do with %s = %s',
                             word, str(score))
                score = None
        else:
            score = None
        if has_neg:
            score = -score
        return score

    def f_nrc_project_lexicon(self, s):
        n_pos = 0
        n_neg = 0
        total_pos = 0
        total_neg = 0
        max_pos = 0
        max_neg = 0
        last_scored_token = 0
        last_token_score = 0
        for item in self.get_items(s):
            item = item.lower()
            score = 0
            if item in self.lexicon:
                score = self.word_to_score(item)
                if score is None:
                    continue
                if score > 0:
                    n_pos += 1
                    total_pos += score
                    if score > max_pos:
                        max_pos = score
                else:
                    n_neg += 1
                    score = -score
                    total_neg += score
                    if score > max_neg:
                        max_neg = score
            if score != 0:
                last_scored_token = score
        last_token_score = self.word_to_score(s[-1])
        if last_token_score is None:
            last_token_score = 0
        return [n_pos,
                n_neg,
                total_pos,
                total_neg,
                total_pos - total_neg,
                max_pos,
                max_neg,
                last_scored_token,
                last_token_score]
    __call__ = f_nrc_project_lexicon


## Input Modifiers (IM)
class IM_Project_Lexicon(object):
    def __init__(self, lexicon, not_found='NOT_FOUND'):
        self.lexicon = lexicon
        self.not_found = not_found

    def im_project_lexicon(self, s):
        ret = []
        for word in s.split(' '):
            word = word.lower()
            if word in self.lexicon:
                ret.append(self.lexicon[word])
            else:
                ret.append(self.not_found)
        return ' '.join(ret)
    __call__ = im_project_lexicon


def im_neg_context(s):
    """Return an enriched representation of the input string with negated
contexts.

    Add _NEG to words in negated contexts (see
    http://sentiment.christopherpotts.net/lingstruc.html).

    Args:
        s: A string.

    Returns:
        An enriched representation of the input string with negation contexts.

    """
    re_beg_ctxt = r"(\b(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b|n't\b)"
    re_end_ctxt = r"[.,:;!?]+"
    s_with_neg = []
    in_neg_ctxt = False
    for word in s.split(' '):
        if re.search(re_end_ctxt, word, flags=re.IGNORECASE):
            in_neg_ctxt = False
        elif re.search(re_beg_ctxt, word, flags=re.IGNORECASE):
            in_neg_ctxt = True
        elif in_neg_ctxt:
            word = word + '_NEG'
        s_with_neg.append(word)
    return ' '.join(s_with_neg)


##### Tokenizer
def happyfuntokenizer(s, preserve_case=True):
    """Tokenize a string with happyfuntokenizing.py.

    Args:
        s: A string to tokenize.

    Returns:
        The string tokenized.
    """
    tok = happyfuntokenizing.Tokenizer(preserve_case=preserve_case)
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


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


def dummy(*_):
    """Dummy function"""
    return 'dummy'


def string_to_feature(s, prefix):
    f = {}
    for (idx, val) in enumerate(s.split()):
        f['%s-%d' % (prefix, idx)] = val
    return f


class String_To_Feature(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def string_to_feature(self, s):
        f = {}
        for (idx, val) in enumerate(s.split()):
            f['%s-%d' % (self.prefix, idx)] = val
        return f
    __call__ = string_to_feature


class DropOn(object):
    """Drop all elements for all keys in the input dictionnary based on
the key and elements associated to key to drop.
    """
    def __init__(self, drop_on, drop):
        self.drop_on = drop_on
        self.drop = drop

    def drop_on(self, d):
        idxes = []
        # Extract all index of values to drop
        for (idx, el) in enumerate(d[self.drop_on].split(' ')):
            if el in self.drop:
                idxes.append(idx)
        # For each element of the input dict
        for key in d:
            new_val = []
            # Drop all values based on the indexes extracted previously
            for (idx, el) in enumerate(d[key].split(' ')):
                if idx not in idxes:
                    new_val.append(el)
            d[key] = ' '.join(new_val)
        return d
    __call__ = drop_on


class KeepOn(object):
    """Keep all elements for all keys in the input dictionnary based on
the key and elements associated to key to keep.
    """
    def __init__(self, keep_on, keep):
        self.keep_on = keep_on
        self.keep = keep

    def keep_on(self, d):
        idxes = []
        # Extract all index of values to keep
        for (idx, el) in enumerate(d[self.keep_on].split(' ')):
            if el in self.keep:
                idxes.append(idx)
        # For each element of the input dict
        for key in d:
            new_val = []
            # Keep all values based on the indexes extracted previously
            for (idx, el) in enumerate(d[key].split(' ')):
                if idx in idxes:
                    new_val.append(el)
            d[key] = ' '.join(new_val)
        return d
    __call__ = keep_on


def build_ineq(lexicon, output_path, vocab=None, truncate=None):
    """Write inequalities for SWE like models in output_path.

    For each word of each class in lexicon, build inqualities like
this :
    lexicon['c1'][i] lexicon['c1'][j] lexicon['c1'][i] lexicon['c2'][l]

    If vocab is supplied, do not build inequalities for words that
aren't in the vocabulary.
    """
    with codecs.open(output_path, 'w+', 'utf-8') as ofile:
        lexicon_inv = utils.invert_dict_nonunique(lexicon)
        for (c1, c2) in itertools.combinations(lexicon_inv, 2):
            lst_c1 = lexicon_inv[c1]
            if vocab is not None:
                for w in lst_c1:
                    if w not in vocab:
                        lst_c1.remove(w)
            random.shuffle(lst_c1)
            lst_c1 = lst_c1[:truncate]

            lst_c2 = lexicon_inv[c2]
            if vocab is not None:
                for w in lst_c2:
                    if w not in vocab:
                        lst_c2.remove(w)
            random.shuffle(lst_c2)
            lst_c2 = lst_c2[:truncate]

            for (c1_w1, c1_w2) in itertools.combinations(lst_c1, 2):
                for c2_w1 in lst_c2:
                    ofile.write('%s %s %s %s\n' % (c1_w1, c1_w2,
                                                   c1_w1, c2_w1))


def find_ineq(model, lexicon, truncate=None):
    lexicon_inv = utils.invert_dict_nonunique(lexicon)
    ineq = []
    cpt = 0
    for (c1, c2) in itertools.combinations(lexicon_inv, 2):
        # c2, c1 = c1, c2
        # All vectors of words in c1
        c1_w = [w for w in lexicon_inv[c1][:truncate] if w in model]
        c1_v = np.array([model[w] for w in c1_w])
        # All vectors of words in c2
        c2_w = [w for w in lexicon_inv[c2][:truncate] if w in model]
        c2_v = np.array([model[w] for w in c2_w])

        logger.info('c1_v = %s', c1_v.shape)
        logger.info('c2_v = %s', c2_v.shape)
        # Concatenate c1 and c2 vectors
        c1_c2_v = np.append(c1_v, c2_v, 0)

        # The index of the first word of c2 in c1_c2_v
        first_c2_i = c1_v.shape[0]

        # The distances between each words in c1 to c1 and c2
        cdist = scipy.spatial.distance.cdist(c1_v, c1_c2_v, metric='cosine')
        # We only consider the strict upper triangle of the cdist
        # matrix, everything under the diagonal have already been
        # considered
        cdist = np.triu(cdist, 1)

        # The index of words distances ordered
        sorted_cdist_idx = np.argsort(cdist)

        # Iter on rows (words of c1)
        for i in range(c1_v.shape[0]):
            if i % 10 == 0:
                logger.info('%d/%d = %d', i, c1_v.shape[0], cpt)
            start_recording = False
            c2_to_reorder = []
            c1_to_reorder = []
            # Iter on columns (ordered index of dist(c1_v[i], c2_v))
            for (idx_j, j) in enumerate(sorted_cdist_idx[i][i:]):
                # If the current index is in c2
                if j >= first_c2_i:
                    # Star recording the next c1 words because they
                    # should be before j
                    start_recording = True
                    # Store the c2 to reorder and the position from
                    # which the next c1 words should be reordered
                    c2_to_reorder.append([j, len(c1_to_reorder)])
                # The current index is in c1
                elif start_recording:
                    # But there are c2 words before, store the c1
                    # words that have to be reordered
                    c1_to_reorder.append(j)
            # Columns is over, generate the inequalities to reorder c1
            # words before c2 words

            # The word to where everything started (to which the
            # columns was built)
            c1_w1 = c1_w[i]
            # c1_w1 = i
            # For all c2 words that have to be reordered
            for (j, start_pos) in c2_to_reorder:
                # The current c2 word to reorder realigned
                c2_w1 = c2_w[j - first_c2_i - 1]
                # c2_w1 = j - first_c2_i - 1
                # For all c1 words that have to be reordered before
                # the current c2 word to reorder
                for k in c1_to_reorder[:start_pos]:
                    c1_w2 = c1_w[k]
                    # c1_w2 = k
                    # The inequality is made of c1_w1, the row word,
                    # and c1_w2, the c1 word after c2_w1, and c2_w1,
                    # the c2 word before c1_w2
                    # ineq.append([c1_w1, c1_w2, c2_w1])
                    cpt += 1
        logger.info('total = %d', cpt)
        return ineq
