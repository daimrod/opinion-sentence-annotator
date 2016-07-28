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
from sklearn.grid_search import GridSearchCV

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

parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': [True, False],
              'clf__alpha': [1e-2, 1e-3, 1e-4, 1e-5],
              'clf__n_iter': [1, 2, 5, 10],
              'clf__loss': ['hinge', 'log', 'perceptron'],
}
pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge')),
])

gs_clf = GridSearchCV(pipeline, parameters, n_jobs=6)
gs_clf = gs_clf.fit(train.data, train.target)
for param in gs_clf.best_params_:
    print('%s: %r' % (param, gs_clf.best_params_[param]))

predicted = gs_clf.predict(test.data)
print(metrics.classification_report(test.target, predicted,
                                    target_names=labels))

