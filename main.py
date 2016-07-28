#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import data
import convert
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
import numpy as np

logger = logging.getLogger(__name__)


train = convert.read_dataset(data.semeval16_polarity_train)
# Convert objective and neutral to objective/neutral
convert.merge_classes(train.target_names,
                       ['objective',
                        'neutral',
                        'objective-OR-neutral'],
                       'neutral')
# Build the target array
target, labels = convert.strings_to_integers(train.target_names)
train.target.extend(target)

test = convert.read_dataset(data.semeval16_polarity_test)
# Convert objective and neutral to objective/neutral
convert.merge_classes(test.target_names,
                       ['objective',
                        'neutral',
                        'objective-OR-neutral'],
                       'neutral')
# Build the target array
target, labels = convert.strings_to_integers(test.target_names)
test.target.extend(target)

text_pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier(loss='hinge',
                                                n_iter=5,
                                                random_state=42)),
])

text_clf = text_pipeline.fit(train.data, train.target)
predicted = text_clf.predict(test.data)
print(metrics.classification_report(test.target, predicted,
                                    target_names=labels))

## DEPRECATED


def old_pipeline(train):
    # Tokenizing
    count_vec = CountVectorizer()
    X_train_counts = count_vec.fit_transform(train.data)

    tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, train.target)

    docs_new = ['I\'m sad.', 'I\'m happy']
    X_new_counts = count_vec.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, train.target_names[category]))
