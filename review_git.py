# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 23:44:39 2017

@author: Siva Kanishka
"""

import numpy as np
import pandas as pd
import re
import nltk

# Reading dataset
raw_pd = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Converting text data structure to list
raw = list(raw_pd['Review'])

from nltk.tokenize import sent_tokenize, word_tokenize

# Removing special characters except apostrophes
def clean(text, keep_apos = False):
    if keep_apos:
        pattern = r'[!|@|#|$|%|&|*|?|(|)|~]'
        filtered_text = re.sub(pattern, r'', text)
    else:
        pattern = r'[^a-zA-Z]'
        filtered_text = re.sub(pattern, r'', text)
    return filtered_text

# Expanding contractions
from contractions import CONTRACTION_MAP

def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

# Removing stop words and stemming
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    text = raw[i]
    review = clean(text, keep_apos = True)
    review.lower()
    review = expand_contractions(review, CONTRACTION_MAP)
    review = clean(review, keep_apos = False)
    review = review.split()
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    stop_words.remove('no')
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = raw_pd.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from __future__ import division
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# 
accuracy = (cm[1,1] + cm[0,0])/(sum(sum(cm)))
precision = cm[1,1]/(cm[0,1] + cm[1,1])
recall = cm[1,1]/(cm[1,0] + cm[1,1])
f1 = (2*precision*recall)/(precision+recall)





















