#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import codecs
from collections import namedtuple
from subprocess import Popen, PIPE

import tempfile

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau

import happyfuntokenizing

from gensim import models

from nltk.tokenize import TweetTokenizer


logger = logging.getLogger(__name__)

## Configuration
import os

if 'DATA_DIR' in os.environ:
    DATA_DIR = os.environ['DATA_DIR']
else:
    DATA_DIR = '/media/jadi-g/DATADRIVE1/corpus'

semeval13 = os.path.join(DATA_DIR, 'SEMEVAL13')
semeval13_polarity_train = os.path.join(semeval13, 'tweeti-b.dist.tsv.2')
semeval13_polarity_dev = os.path.join(semeval13, 'tweeti-b.dev.dist.tsv.2')
semeval16 = os.path.join(DATA_DIR, 'SEMEVAL16')
semeval16_polarity_train = os.path.join(semeval16,
                                        'Task4',
                                        '100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt')
semeval16_polarity_test = os.path.join(semeval16,
                                       'Task4',
                                       '100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt')

SENNA_PATH = os.path.expanduser('~/src/thesis/senna/')

########## Data Reader
Dataset = namedtuple('Dataset', ['data',
                                 'filenames',  # we don't need this for semeval
                                 'target_names',
                                 'target',
                                 'uid',  'sid',  # we need those for semeval
                                 ])


def read_dataset(ipath, separator='\t',
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


########## Emebeddings
def Compare_ANEW(file_opinion):
    h_word2valence, h_word2arousal, h_word2dominance = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            l = l.rstrip('\r\n')
            w, nb, valence, _, arousal, _, dominance, _ = l.split('\t')
            if w+'___N' in h_word2dim:
                w += '___N'
            elif w+'___V' in h_word2dim:
                w += '___V'
            elif w+'___A' in h_word2dim:
                w += '___A'
            else:
                continue

            dim_w = h_word2dim[w]
            if dim_w >= N:
                continue
            h_word2valence[w] = float(valence)
            h_word2dominance[w] = float(dominance)
            h_word2arousal[w] = float(arousal)

    topn, p = 1500, 1
    t_r, t_rho, t_tau = [], [], []
    for w in h_word2valence:
        valence = h_word2valence[w]
        dominance = h_word2dominance[w]
        arousal = h_word2arousal[w]

        # closest words
        #t_closest_valence = sorted(h_word2valence.keys(),key=lambda v:abs(valence-h_word2valence[v]))[0:topn]
        #t_closest_dominance = sorted(h_word2dominance.keys(),key=lambda v:abs(dominance-h_word2dominance[v]))[0:topn]
        #t_closest_arousal = sorted(h_word2arousal.keys(),key=lambda v:abs(arousal-h_word2arousal[v]))[0:topn]

        h_treeD = {}
        for v in list(h_word2arousal):
            h_arousal = abs(arousal-h_word2arousal[v])**p
            h_valence = abs(valence-h_word2valence[v])**p
            h_dominance = abs(dominance-h_word2dominance[v])**p
            h_threeD[v] = (h_arousal + h_valence + h_dominance)**(1/p)

        t_closest_threeD = sorted(list(h_threeD),
                                  key=lambda w2: (h_threeD[w2],
                                                  -m_embed[h_word2dim[w],
                                                           h_word2dim[w2]]))
        # t_closest_threeD = t_closest_threeD[::60]

        # random
        #t_random_valence = sorted(h_word2valence.keys(),key=lambda v:abs(valence-h_word2valence[v]))[0:topn]
        #t_random_dominance = sorted(h_word2dominance.keys(),key=lambda v:abs(dominance-h_word2dominance[v]))[0:topn]
        #t_random_arousal = sorted(h_word2arousal.keys(),key=lambda v:abs(arousal-h_word2arousal[v]))[0:topn]
        #t_random_threeD = sorted(h_word2arousal.keys(),key=lambda v:sqrt(abs(arousal-h_word2arousal[v])**2 + abs(valence-h_word2valence[v])**2 + abs(dominance-h_word2dominance[v])**2 ) )[0:topn]

        #t_valence_ordered = [h_word2valence[w2] for w2 in t_closest_valence]
        t_threeD_ordered = [h_threeD[w2] for w2 in t_closest_threeD]

        t_NN_ordered = [-m_embed[h_word2dim[w], h_word2dim[w2]]
                        for w2 in t_closest_threeD]

        t_r.append(scipy.stats.stats.pearsonr(t_threeD_ordered,
                                              t_NN_ordered)[0])
        t_rho.append(scipy.stats.stats.spearmanr(t_threeD_ordered,
                                                 t_NN_ordered)[0])
        t_tau.append(scipy.stats.stats.kendalltau(t_threeD_ordered,
                                                  t_NN_ordered)[0])

    print('number of words processed: ', len(t_r))
    print('Mean pearson r =', sum(t_r)/len(t_r))
    print('Mean spearman rho =', sum(t_rho)/len(t_rho))
    print('Mean kendall tau =', sum(t_tau)/len(t_tau))


def Compare_SimLexANEW(file_simlex, file_opinion):

    h_convert_POS = {'A': 'ADJ', 'N': 'N', 'V': 'V'}
    h_SimLex, h_voc = {}, {}
    for line_no, line in enumerate(codecs.open(file_simlex, 'r', 'utf-8')):
        if line_no == 0:
            continue
        t_line = line.split('\t')
        w1 = t_line[0]
        w2 = t_line[1]
        POS = t_line[2]
        simlex999 = t_line[3]
        bool_sim333 = t_line[8]

        POS = h_convert_POS[POS]
        w1, w2 = w1+'___'+POS, w2+'___'+POS
        simlex999 = float(simlex999)
        h_SimLex[w1+'###'+w2] = simlex999
        h_voc[w1] = 1
        h_voc[w2] = 1

    h_word2valence, h_word2arousal, h_word2dominance = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            l = l.rstrip('\r\n')
            w, nb, valence, _, arousal, _, dominance, _ = l.split('\t')
            if w+'___N' in h_word2dim:
                w += '___N'
            elif w+'___V' in h_word2dim:
                w += '___V'
            elif w+'___A' in h_word2dim:
                w += '___A'
            else:
                continue

            dim_w = h_word2dim[w]
            if dim_w >= N:
                continue
            h_word2valence[w] = float(valence)
            h_word2dominance[w] = float(dominance)
            h_word2arousal[w] = float(arousal)

    t_opinion_ordered, t_simlex_ordered = [], []
    for tw in h_SimLex:
        w1, w2 = tw.split('###')
        if w1 in h_word2valence and w2 in h_word2valence:
            t_opinion_ordered.append(abs(h_word2valence[w1]
                                         - h_word2valence[w2]))
            t_simlex_ordered.append(h_SimLex[tw])

        #valence, dominance, arousal = h_word2valence[w],h_word2dominance[w],h_word2arousal[w]

    print('Pearson r SimLex/ANEW = ', pearsonr(t_opinion_ordered,
                                               t_simlex_ordered)[0])
    print('Spearman rho SimLex/ANEW = ', spearmanr(t_opinion_ordered,
                                                   t_simlex_ordered)[0])
    print('Kendall tau SimLex/ANEW = ', kendalltau(t_opinion_ordered,
                                                   t_simlex_ordered)[0])


def Compare_SentiWN(file_opinion):

    h_word2positive, h_word2negative, h_word2objective = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            t, _, pos, neg, t_w, _ = l.rstrip('\r\n').split('\t')

            for w in t_w.split(' '):
                w = re.sub('#.+', '', w)

                if t == 'n' and w+'___N' in h_word2dim:
                    w += '___N'
                elif t == 'v' and w+'___V' in h_word2dim:
                    w += '___V'
                elif t == 'r' and w+'___ADV' in h_word2dim:
                    w += '___ADV'
                elif t == 'a' and w+'___A' in h_word2dim:
                    w += '___A'
                else:
                    continue

                dim_w = h_word2dim[w]
                if dim_w >= N:
                    continue

                h_word2positive[w] = float(pos)
                h_word2negative[w] = float(neg)
                h_word2objective[w] = 1 - (float(pos)+float(neg))

    print('nb mots de sentiWN ds embed ', len(h_word2objective))

    topn, p = 180, 1
    t_r, t_rho, t_tau = [], [], []
    for w in random.sample(list(h_word2positive), 2000):
        objective = h_word2objective[w]
        positive = h_word2positive[w]
        negative = h_word2negative[w]

        # closest words

        #h_threeD = { v:(abs(positive-h_word2positive[v])**p + abs(negative-h_word2negative[v])**p )**(1/p) for v in list(h_word2positive) }
        h_threeD = {v: abs(negative-h_word2negative[v])
                    for v in h_word2positive}

        t_closest_threeD = sorted(list(h_threeD),
                                  key=lambda w2: (h_threeD[w2],
                                                  -m_embed[h_word2dim[w],
                                                           h_word2dim[w2]]))
        t_closest_threeD = t_closest_threeD[::topn]

        # random
        #t_random_valence = sorted(h_word2valence.keys(),key=lambda v:abs(valence-h_word2valence[v]))[0:topn]
        #t_random_dominance = sorted(h_word2dominance.keys(),key=lambda v:abs(dominance-h_word2dominance[v]))[0:topn]
        #t_random_arousal = sorted(h_word2arousal.keys(),key=lambda v:abs(arousal-h_word2arousal[v]))[0:topn]
        #t_random_threeD = sorted(h_word2arousal.keys(),key=lambda v:sqrt(abs(arousal-h_word2arousal[v])**2 + abs(valence-h_word2valence[v])**2 + abs(dominance-h_word2dominance[v])**2 ) )[0:topn]

        #t_valence_ordered = [h_word2valence[w2] for w2 in t_closest_valence]
        t_threeD_ordered = [h_threeD[w2] for w2 in t_closest_threeD]

        t_NN_ordered = [-m_embed[h_word2dim[w], h_word2dim[w2]]
                        for w2 in t_closest_threeD]

        t_r.append(pearsonr(t_threeD_ordered, t_NN_ordered)[0])
        t_rho.append(spearmanr(t_threeD_ordered, t_NN_ordered)[0])
        t_tau.append(kendalltau(t_threeD_ordered, t_NN_ordered)[0])

    print('number of words processed: ', len(t_r))
    print('Mean Pearson r =', sum(t_r)/len(t_r))
    print('Mean Spearman rho =', sum(t_rho)/len(t_rho))
    print('Mean Kendall tau =', sum(t_tau)/len(t_tau))


def Compare_SimLexSentiWN(file_simlex, file_opinion):

    h_convert_POS = {'A': 'ADJ', 'N': 'N', 'V': 'V'}
    h_SimLex, h_voc = {}, {}
    for line_no, line in enumerate(codecs.open(file_simlex, 'r', 'utf-8')):
        if line_no == 0:
            continue
        t_line = line.split('\t')
        w1 = t_line[0]
        w2 = t_line[1]
        POS = t_line[2]
        simlex999 = t_line[3]
        bool_sim333 = t_line[8]

        POS = h_convert_POS[POS]
        w1, w2 = w1+'___'+POS, w2+'___'+POS
        simlex999 = float(simlex999)
        h_SimLex[w1+'###'+w2] = simlex999
        h_voc[w1] = 1
        h_voc[w2] = 1

    h_word2positive, h_word2negative, h_word2objective = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            t, _, pos, neg, t_w, _ = l.rstrip('\r\n').split('\t')

            for w in t_w.split(' '):
                w = re.sub('#.+', '', w)

                if t == 'n' and w+'___N' in h_word2dim:
                    w += '___N'
                elif t == 'v' and w+'___V' in h_word2dim:
                    w += '___V'
                elif t == 'r' and w+'___ADV' in h_word2dim:
                    w += '___ADV'
                elif t == 'a' and w+'___A' in h_word2dim:
                    w += '___A'
                else:
                    continue

                dim_w = h_word2dim[w]
                if dim_w >= N:
                    continue

                h_word2positive[w] = float(pos)
                h_word2negative[w] = float(neg)
                h_word2objective[w] = 1 - (float(pos)+float(neg))

    print('nb mots de sentiWN ds SimLex ', len(h_word2objective))

    t_opinion_ordered, t_simlex_ordered = [], []
    for tw in h_SimLex:  # sorted(list(h_SimLex),key=h_SimLex.get)[::10]:
        w1, w2 = tw.split('###')
        if w1 in h_word2objective and w2 in h_word2objective:
            t_opinion_ordered.append(abs(h_word2negative[w1]
                                         - h_word2negative[w2]))
            t_simlex_ordered.append(h_SimLex[tw])

    print('size of intersection: ', len(t_opinion_ordered))
    print('Pearson r SimLex/SentiWN = ', pearsonr(t_opinion_ordered,
                                                  t_simlex_ordered)[0])
    print('Spearman rho SimLex/SentiWN = ', spearmanr(t_opinion_ordered,
                                                      t_simlex_ordered)[0])
    print('Kendall tau SimLex/SentiWN = ', kendalltau(t_opinion_ordered,
                                                      t_simlex_ordered)[0])


def Compare_ANEWSentiWN(file_anew, file_sentiwn):

    h_word2positive, h_word2negative, h_word2objective = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            t, _, pos, neg, t_w, _ = l.rstrip('\r\n').split('\t')

            for w in t_w.split(' '):
                w = re.sub('#.+', '', w)

                if t == 'n' and w+'___N' in h_word2dim:
                    w += '___N'
                elif t == 'v' and w+'___V' in h_word2dim:
                    w += '___V'
                elif t == 'r' and w+'___ADV' in h_word2dim:
                    w += '___ADV'
                elif t == 'a' and w+'___A' in h_word2dim:
                    w += '___A'
                else:
                    continue

                dim_w = h_word2dim[w]
                if dim_w >= N:
                    continue

                h_word2positive[w] = float(pos)
                h_word2negative[w] = float(neg)
                h_word2objective[w] = 1 - (float(pos)+float(neg))

    h_word2valence, h_word2arousal, h_word2dominance = {}, {}, {}
    with open(file_anew) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            l = l.rstrip('\r\n')
            w, nb, valence, _, arousal, _, dominance, _ = l.split('\t')
            if w+'___N' in h_word2dim:
                w += '___N'
            elif w+'___V' in h_word2dim:
                w += '___V'
            elif w+'___A' in h_word2dim:
                w += '___A'
            else:
                continue

            h_word2valence[w] = float(valence)
            h_word2dominance[w] = float(dominance)
            h_word2arousal[w] = float(arousal)

    print('nb mots de sentiWN ds ANEW ', len(h_word2valence))

    t_word = sorted(list(h_word2valence), key=lambda w: h_word2valence[w])
    t_opinion1_ordered = [h_word2valence[w] for w in t_word]
    t_opinion2_ordered = [h_word2positive[w]-h_word2negative[w]
                          for w in t_word]

    print('size of intersection: ', len(t_opinion1_ordered))
    print('Pearson r SimLex/SentiWN = ', pearsonr(t_opinion1_ordered,
                                                  t_opinion2_ordered)[0])
    print('Spearman rho SimLex/SentiWN = ', spearmanr(t_opinion1_ordered,
                                                      t_opinion2_ordered)[0])
    print('Kendall tau SimLex/SentiWN = ', kendalltau(t_opinion1_ordered,
                                                      t_opinion2_ordered)[0])


def Compare_W2V_ANEW(file_opinion):
     model_w2v = models.Word2Vec.load_word2vec_format('/home/vincent/Data/Data/Google_Word2Vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

     h_word2valence, h_word2arousal, h_word2dominance = {}, {} ,{}
     with open(file_opinion) as f:
         for l in f:
             if l.startswith('Word') or l.startswith('#'): continue
             w,nb,valence,_,arousal,_,dominance,_ = l.rstrip('\r\n').split('\t')
             if w not in model_w2v: continue

             h_word2valence[w]=float(valence)
             h_word2dominance[w]=float(dominance)
             h_word2arousal[w]=float(arousal)


     topn, p = 1500, 1
     t_r, t_rho, t_tau = [],[],[]
     for w in h_word2valence:
         valence, dominance, arousal = h_word2valence[w],h_word2dominance[w],h_word2arousal[w]

         h_threeD = { v:(0*abs(arousal-h_word2arousal[v])**p + 0*abs(valence-h_word2valence[v])**p + 1*abs(dominance-h_word2dominance[v])**p )**(1/p) for v in list(h_word2arousal) }

         t_closest_threeD = sorted( list(h_threeD), key= lambda w2: (h_threeD[w2],-model_w2v.similarity(w,w2) ))#[::60]

         t_threeD_ordered = [ h_threeD[w2] for w2 in t_closest_threeD ]

         t_NN_ordered = [ -model_w2v.similarity(w,w2) for w2 in t_closest_threeD ]

         t_r.append( scipy.stats.stats.pearsonr(t_threeD_ordered, t_NN_ordered)[0] )
         t_rho.append( scipy.stats.stats.spearmanr(t_threeD_ordered, t_NN_ordered)[0] )
         t_tau.append( scipy.stats.stats.kendalltau(t_threeD_ordered , t_NN_ordered)[0] )

     print('number of words processed: ', len(t_r))
     print('Mean pearson r =', sum(t_r)/len(t_r))
     print('Mean spearman rho =', sum(t_rho)/len(t_rho))
     print('Mean kendall tau =', sum(t_tau)/len(t_tau))


def Compare_W2V_SentiWN(file_opinion):
     from gensim import models
     model_w2v = 
models.Word2Vec.load_word2vec_format('/home/vincent/Data/Data/Google_Word2Vec/GoogleNews-vectors-negative300.bin.gz', 
binary=True)

     h_word2positive, h_word2negative, h_word2objective = {}, {} ,{}
     with open(file_opinion) as f:
         for l in f:
             if l.startswith('Word') or l.startswith('#'): continue
             t,_,pos,neg,t_w,_ = l.rstrip('\r\n').split('\t')

             for w in t_w.split(' '):
                 w = re.sub('#.+','',w)

                 if w not in model_w2v: continue

                 h_word2positive[w] = float(pos)
                 h_word2negative[w] = float(neg)
                 h_word2objective[w]= 1 - (float(pos)+float(neg))

     print('nb mots de sentiWN ds W2V ',len(h_word2objective))


     topn, p = 180, 1
     t_r, t_rho, t_tau = [],[],[]
     for w in random.sample(list(h_word2positive),2000):
         objective, positive, negative = 
h_word2objective[w],h_word2positive[w],h_word2negative[w]

         # closest words

         #h_threeD = { v:(abs(positive-h_word2positive[v])**p + 
abs(negative-h_word2negative[v])**p )**(1/p) for v in 
list(h_word2positive) }
         h_threeD = { v:abs(objective-h_word2objective[v]) for v in 
h_word2positive }

         t_closest_threeD = sorted( list(h_threeD), key=lambda w2: 
(h_threeD[w2],-model_w2v.similarity(w,w2) ))[::topn]

         # random
         #t_random_valence = sorted(h_word2valence.keys(),key=lambda 
v:abs(valence-h_word2valence[v]))[0:topn]
         #t_random_dominance = sorted(h_word2dominance.keys(),key=lambda 
v:abs(dominance-h_word2dominance[v]))[0:topn]
         #t_random_arousal = sorted(h_word2arousal.keys(),key=lambda 
v:abs(arousal-h_word2arousal[v]))[0:topn]
         #t_random_threeD = sorted(h_word2arousal.keys(),key=lambda 
v:sqrt(abs(arousal-h_word2arousal[v])**2 + 
abs(valence-h_word2valence[v])**2 + 
abs(dominance-h_word2dominance[v])**2 ) )[0:topn]


         #t_valence_ordered = [h_word2valence[w2] for w2 in 
t_closest_valence]
         t_threeD_ordered = [ h_threeD[w2] for w2 in t_closest_threeD ]

         t_NN_ordered = [ -model_w2v.similarity(w,w2) for w2 in 
t_closest_threeD ]

         t_r.append( scipy.stats.stats.pearsonr(t_threeD_ordered, 
t_NN_ordered)[0] )
         t_rho.append( scipy.stats.stats.spearmanr(t_threeD_ordered, 
t_NN_ordered)[0] )
         t_tau.append( scipy.stats.stats.kendalltau(t_threeD_ordered , 
t_NN_ordered)[0] )

     print('number of words processed: ', len(t_r))
     print('Mean Pearson r =', sum(t_r)/len(t_r))
     print('Mean Spearman rho =', sum(t_rho)/len(t_rho))
     print('Mean Kendall tau =', sum(t_tau)/len(t_tau))



def BuildQREL_from_NRC_W2V(file_opinion,model_w2v):
     Info('NRC with W2V')

     h_word2class, h_class2word = defaultdict(lambda:[]), 
defaultdict(lambda:[])
     # ordre des classes: anger,anticipation, disgust, fear,joy, 
negative, positive, sadness, surprise, trust
     with open(file_opinion) as f:
         for l in f:
             t_l = l.rstrip('\r\n').split('\t')
             if len(t_l)==3:
                 w,v = t_l[0],int(t_l[2])

                 if w+'_N' in model_w2v: w+='_N'
                 elif w+'_V' in model_w2v: w+='_V'
                 elif w+'_A' in model_w2v: w+='_A'
                 else:
                     # si on utilise les vecteur de Google News
                     if w not in model_w2v: continue


                 #if t_l[1] not in ['positive','negative']: continue
                 if t_l[1] not in ['sadness']: continue
                 h_word2class[w].append(str(v))
                 h_class2word[t_l[1]].append(w)

     f_qrel_anger = codecs.open(args.prefix+'.NRC_all_dim.qrel','w')
     qid, h_qid2w = 1, {}
     h_seen = {}
     for w in list(h_word2class)[0:5000]: # 1000 requetes suffisent
         vector = ''.join(h_word2class[w])
         if w in h_seen or vector == '0': continue
         #if w in h_seen or vector == '0000000000': continue
         #if w in h_seen or vector == '00': continue
         t_pos_w = [w]
         t_identical = [ w2 for w2 in h_word2class if 
''.join(h_word2class[w2])==vector and not w2==w and not w2 in h_seen ]
         #t_pos_w += random.sample(t_identical,min(len(t_identical),1))
         for w2 in t_pos_w: h_seen[w2]=1
         t_neg_w = []
         #s_anti_w = set([ w2 for w2 in h_word2class if not 
''.join(h_word2class[w2]) == vector and not 
''.join(h_word2class[w2])=='00'])
         #for t_elem in MostSimilar([w],m_embed,h_word2dim, topn=200):#, 
possible_tag='same_as_query'):
         #    nn,score = t_elem
         #    if nn in s_anti_w:
         #        t_neg_w.append(nn)
         #        if len(t_neg_w)>0: break
         #print(w, '  ;  neg : '+' '.join(t_neg_w))


         for nn in ( w2 for w2 in h_word2class if vector == 
(''.join(h_word2class[w2])) ):
             if nn in t_pos_w: continue
             f_qrel_anger.write(str(qid)+' 0 '+nn+' 1\n')

         h_qid2w[qid]=(t_pos_w,t_neg_w)
         qid+=1
     f_qrel_anger.close()
     return h_qid2w, h_word2class

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


##### Caps
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
        if word.upper() == word:
            n = n + 1
    return n


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

    d = {'POS': [], 'CHK': [], 'NER': []}
    for line in out.split('\n'):
        line = line.strip()
        if not line:
            if d['POS'] != []:
                ret.append(d)
                d = {'POS': [], 'CHK': [], 'NER': []}
            continue
        tags = line.split('\t')
        tags = [x.strip() for x in tags]
        pos, chk, ner = tags
        d['POS'].append(pos)
        d['CHK'].append(chk)
        d['NER'].append(ner)
        new_line = False
    return ret


##### Tokenizer
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


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


def dummy(*args):
    return 'dummy'


class Tokenizer(BaseEstimator, TransformerMixin):
    """Tokenize with the given tokenizer."""
    def __init__(self, item=None, tokenizer=None):
        self.item = item
        self.tokenizer = tokenizer
        self._update_params()

    def _update_params(self):
        if self.tokenizer is None:
            self.tokenizer = identity

    def fit(self, X, y=None, *params):
        return self

    def transform(self, X, *params):
        if self.item is None:
            return [self.tokenizer(x) for x in X]
        else:
            new_X = []
            for x in X:
                new_x = dict(x)
                new_x[self.item] = self.tokenizer(x[self.item])
                new_X.append(new_x)
        return new_X

    def get_params(self, deep=True):
        return {'item': self.item,
                'tokenizer': self.tokenizer}

    def set_params(self, **params):
        for p in params:
            setattr(self, p, params[p])
        self._update_params()


class Filter(BaseEstimator, TransformerMixin):
    """Filter input based."""
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None, *params):
        return self

    def transform(self, X, *params):
        if self.enabled:
            return X
        else:
            return [[0] for x in X]

    def get_params(self, deep=True):
        return {'enabled': self.enabled}

    def set_params(self, **params):
        for p in params:
            setattr(self, p, params[p])


########## Pipeline
def run(truncate=None):
    train = read_dataset(semeval16_polarity_train)
    # Convert objective and neutral to objective/neutral
    merge_classes(train.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    # Build the target array
    target, labels = strings_to_integers(train.target_names)
    train.target.extend(target)

    test = read_dataset(semeval16_polarity_test)
    # Convert objective and neutral to objective/neutral
    merge_classes(test.target_names,
                  ['objective',
                   'neutral',
                   'objective-OR-neutral'],
                  'neutral')
    # Build the target array
    target, labels = strings_to_integers(test.target_names)
    test.target.extend(target)

    # parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #               'tfidf__use_idf': [True, False],
    #               'clf__alpha': [1e-3, 1e-4, 1e-5],
    #               'clf__n_iter': [1, 2, 5],
    #               'clf__loss': ['hinge'],
    # }

    parameters = {'tok__tokenizer': [None, happyfuntokenizer, nltktokenizer],

                  'features__ngram__vect__ngram_range': [(1, 3)],
                  'features__ngram__tfidf__use_idf': [True],

                  'clf__alpha': [1e-4],
                  'clf__n_iter': [5],
                  'clf__loss': ['hinge'],
                  }

    pipeline = Pipeline([
        ('tok', Tokenizer('text')),
        ('features', FeatureUnion([
            ('ngram', Pipeline([
                ('extract', ItemExtractor('text')),
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
            ])),
        ])),
        ('clf', SGDClassifier(loss='hinge', random_state=42)),
    ])

    scorer = metrics.make_scorer(metrics.f1_score,
                                 average='micro')
    scorer = metrics.make_scorer(metrics.accuracy_score)
    scorer = 'accuracy'
    clf = GridSearchCV(pipeline, parameters, n_jobs=1,
                       scoring=scorer, verbose=1)
    clf = clf.fit(train.data[:truncate], train.target[:truncate])
    for param in clf.best_params_:
        print('%s: %r' % (param, clf.best_params_[param]))

    # clf = pipeline
    # clf = clf.fit(train.data, train.target)

    predicted = clf.predict(test.data)
    print(metrics.classification_report(test.target, predicted,
                                        target_names=labels))
