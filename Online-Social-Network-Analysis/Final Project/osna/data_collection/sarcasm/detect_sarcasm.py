# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:43:57 2019

@author: alanc
"""

from collections import Counter

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS 

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

def detect_sarcasm(showMode=True):
    print(os.listdir("osna/data_collection/sarcasm/data/"))

    PATH = 'osna/data_collection/'
    PATH_SARCASM = 'osna/data_collection/sarcasm/data/'

    # Get comments
    table = pd.read_csv(PATH+'comments2.csv', sep=';')
    print(table)
    table2 = pd.read_csv(PATH+'comments.csv', sep=';')
    print(table2)
    table3 = pd.read_csv(PATH+'comments3.csv', sep=',')
    print(table3)
    table=table[:1000]
    table2=table2[:1000]
    table3=table3[1001:2000]
    tables=[table, table3,table2]
    table=pd.concat(tables)
    if showMode:
        table.info()

    xpredict=table['body']

    # Get train data from sarcasm labellized data
    train = pd.read_csv(PATH_SARCASM + 'train-balanced-sarcasm.csv')
    # Get test data from sarcasm labellized data
    test = pd.read_csv(PATH_SARCASM + 'test-unbalanced.csv')
    if showMode:
        print(train.head(10))
    train_without_na = train.dropna(subset=['comment'])
    if showMode:
        train_without_na.info()
        train_without_na['label'].value_counts()
        train_without_na['author'].value_counts()
    train_without_na.groupby('author')['label'].agg([np.size, np.mean, np.sum]).sort_values(by='sum', ascending=False)

    if showMode:
        # Wordcloud non-sarcastic
        sarcastic_comments = str(train_without_na[train_without_na['label'] == 1]['comment'])
        plt.figure(figsize=(12, 12))
        word_cloud = WordCloud(stopwords=STOPWORDS)
        word_cloud.generate(sarcastic_comments)
        plt.imshow(word_cloud)

        # Wordcloud sarcastic 
        sincere_comments = str(train_without_na[train_without_na['label'] == 0]['comment'])
        plt.figure(figsize=(12, 12))
        word_cloud = WordCloud(stopwords=STOPWORDS)
        word_cloud.generate(sincere_comments)
        plt.imshow(word_cloud)

    # Analyze data
    train_removed_features = train_without_na.iloc[:, :-3].drop('author', axis=1)
    train_removed_features.head(10)
    train_x, train_y = train_removed_features.drop('label', axis=1), train_removed_features[['label']]
    Counter([word for word in str(train_x['comment']).split(' ') if word not in STOPWORDS])

    # Comment representation with tf/idf
    tfidf_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=None)
    comment_sparse = tfidf_comment.fit_transform(train_x['comment'])
    comment_sparse_pred = tfidf_comment.transform(table['body'])
    print(len(tfidf_comment.vocabulary_))
    x_train = hstack([comment_sparse])
    x_test=hstack([comment_sparse_pred])
    y_train=train_y

    # Logistic Regression
    clf = LogisticRegression(solver='liblinear', verbose=True)
    clf.fit(x_train, y_train)
    if showMode:
        print("accuracy=",clf.score(x_train, y_train))

    # Prediction
    results=clf.predict(x_test)

    table['sarcarsm']=results
    if showMode:
        print(table)
    pd.DataFrame(table).to_csv(PATH + 'commentssarcasm.csv',sep=';')

    return tfidf_comment, clf

if __name__ == '__main__':
    detect_sarcasm()