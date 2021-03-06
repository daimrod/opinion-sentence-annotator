#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

logger = logging.getLogger(__name__)

## Configuration
if 'DATA_DIR' in os.environ:
    DATA_DIR = os.environ['DATA_DIR']
else:
    DATA_DIR = '/media/jadi-g/DATADRIVE1/corpus'

### Stanford Tools
if 'CLASSPATH' not in os.environ:
    os.environ['CLASSPATH'] = ''
os.environ['CLASSPATH'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-parser-full-2015-12-09/') + ':' + os.environ['CLASSPATH']
os.environ['CLASSPATH'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-postagger-full-2015-12-09/') + ':' + os.environ['CLASSPATH']

if 'STANFORD_MODELS' not in os.environ:
    os.environ['STANFORD_MODELS'] = ''
os.environ['STANFORD_MODELS'] = os.path.expanduser('~/src/java/stanford-for-nltk/') + ':' + os.environ['STANFORD_MODELS']
os.environ['STANFORD_MODELS'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-parser-full-2015-12-09/') + ':' + os.environ['STANFORD_MODELS']
os.environ['STANFORD_MODELS'] = os.path.expanduser('~/src/java/stanford-for-nltk/stanford-postagger-full-2015-12-09/models/') + ':' +os.environ['STANFORD_MODELS']


### Dataset
semeval13 = os.path.join(DATA_DIR, 'SEMEVAL13')
semeval13_polarity_train = os.path.join(semeval13, 'tweeti-b.dist.tsv.2')
semeval13_polarity_dev = os.path.join(semeval13, 'tweeti-b.dev.dist.tsv.2')
semeval16 = os.path.join(DATA_DIR, 'SEMEVAL16')
semeval16_polarity_train = os.path.join(semeval16,
                                        'Task4',
                                        '100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt')
semeval16_polarity_devtest = os.path.join(semeval16,
                                       'Task4',
                                       '100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt')

old_rouvier_train = os.path.join(os.path.expanduser('~/src/thesis/Rouvier-SemEval2016/data/'), 'train.txt')
old_rouvier_test = os.path.join(os.path.expanduser('~/src/thesis/Rouvier-SemEval2016/data/'), 'twitter16.txt')
rouvier_train = os.path.join(semeval16, 'Task4', 'rouvier_train.txt')
rouvier_test = os.path.join(semeval16, 'Task4', 'rouvier_twitter16.txt')

# train_path = semeval16_polarity_train
# test_path = semeval16_polarity_devtest
dev_path = semeval16_polarity_devtest
train_path = rouvier_train
test_path = rouvier_test

SENNA_PATH = os.path.expanduser('~/src/thesis/senna/')
SWE_PATH = os.path.expanduser('~/src/thesis/SWE/')
MCE_PATH = os.path.expanduser('~/src/thesis/mce/')

SEMEVAL_SCORER_PATH = os.path.join(semeval16, 'Task4/SemEval2016_task4_submissions_and_scores/')

TREC_EVAL_PATH = os.path.expanduser('~/src/thesis/trec_eval.9.0/')

questions_words_path = os.path.join(DATA_DIR, 'test_data/questions-words.txt')
simlex999_path = os.path.join(DATA_DIR, 'test_data/simlex999.txt')
wordsim353_path = os.path.join(DATA_DIR, 'test_data/wordsim353.tsv')

## Lexicons

### English lexicons
bing_liu_lexicon_path = dict({'negative': os.path.join(DATA_DIR, 'bing-liu-lexicon/negative-words.txt'),
                              'positive': os.path.join(DATA_DIR, 'bing-liu-lexicon/positive-words.txt')})

mpqa_lexicon_path = os.path.join(DATA_DIR, 'subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')

mpqa_plus_lexicon_path = os.path.join(DATA_DIR, 'mpqa_plus_lex.xml')

nrc_emotion_lexicon_path = os.path.join(DATA_DIR, 'NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt.nodoc')
nrc_hashtag_unigram_lexicon_path = os.path.join(DATA_DIR, 'NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt')
nrc_hashtag_bigram_lexicon_path = os.path.join(DATA_DIR, 'NRC-Hashtag-Sentiment-Lexicon-v0.1/bigrams-pmilexicon.txt')
nrc_hashtag_pair_lexicon_path = os.path.join(DATA_DIR, 'NRC-Hashtag-Sentiment-Lexicon-v0.1/pairs-pmilexicon.txt')
nrc_hashtag_sentimenthashtags_lexicon_path = os.path.join(DATA_DIR, 'NRC-Hashtag-Sentiment-Lexicon-v0.1/sentimenthashtags.txt')
nrc_sentiment140_unigram_lexicon_path = os.path.join(DATA_DIR, 'Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt')
nrc_sentiment140_bigram_lexicon_path = os.path.join(DATA_DIR, 'Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt')
nrc_sentiment140_pair_lexicon_path = os.path.join(DATA_DIR, 'Sentiment140-Lexicon-v0.1/pairs-pmilexicon.txt')

### French lexicons
lidilem_base_path = os.path.join(DATA_DIR, 'LIDILEM-Grenoble-lexiques')
lidilem_adjectifs_lexicon_path = os.path.join(lidilem_base_path, 'Adjectifs.csv')
lidilem_noms_lexicon_path = os.path.join(lidilem_base_path, 'Noms.csv')
lidilem_verbes_lexicon_path = os.path.join(lidilem_base_path, 'Verbes.csv')
blogoscopie_lexicon_path = os.path.join(DATA_DIR, 'blogoscopie/francais/LEXIQUE_EVALUATION_INITIAL_STRIPPED.txt')


gnews_negative300_path = os.path.join(DATA_DIR, 'GoogleNews-vectors-negative300.bin.gz')

carnegie_clusters_path = os.path.join(DATA_DIR, 'carnegie_clusters/50mpaths2')

happy_emoticons_path = os.path.join(DATA_DIR, 'happy_emoticons.txt')
sad_emoticons_path = os.path.join(DATA_DIR, 'sad_emoticons.txt')

## Tweets
twitter_logger_en_path = os.path.join(DATA_DIR, 'tweets/en.json')
twitter_logger_fr_path = os.path.join(DATA_DIR, 'tweets/fr.json')
