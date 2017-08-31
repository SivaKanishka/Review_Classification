# Review_Classification
Binary classification of a restaurant reviews

This code uses data from vvvv for analysis

Pre-processing of text data was done using **NLTK** library in Python. The steps of pre-processing were:
1. Removing special characters except apostrophes
2. Contraction of words with apostrophes using a corpus of key-value pairs stored as *contractions.py* file
3. Case conversion of words
4. Second rund of special chracter removal
5. Tokenization of sentences into words
6. Filtering stopwords, excluding negation terms like *not*, stopwords corpus in nltk library
7. Stemming of words using Porter Stemmer

**Bag-of-Words** model was used for feature extraction. feature extraction module from scikit-learn library was used for this. The *hyper parameter* chosen was number of features, with it's value at 500

**Naive Bayes** classifier was trained on the training set after random spliting of the dataset

The following metrics were recorded:
1. Accuracy
2. Precision
3. Recall
4. F1 score
