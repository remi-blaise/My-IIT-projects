#!/usr/bin/python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from random import sample
import pandas 
import numpy as np
from numpy import argmax

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from scipy.sparse import hstack

from seaborn import heatmap
import matplotlib.pyplot as plt


def write_coo_matrix(X_sparse):
    pandas.DataFrame(X_sparse.getcol(-4).toarray()).to_csv('test2.csv')


def get_data():
    # Read data from CSV
    def open_doc(path,sepa):
        return pandas.read_csv(path, sep=sepa)
     
    reader = open_doc("./osna/data_collection/commentssarc.csv",';')
    sentiments = open_doc("./osna/sentiment_analysis/sentiments_usingtextblob.csv",',')

    reader['polarity'] = sentiments['polarity']
    reader['subjectivity'] = sentiments['subjectivity']

    features = ['body', 'polarity', 'subjectivity', 'sarcarsm', 'label']
    data = reader[features]

    # Select as much data for each label
    minus_ones = np.array([ line for line in data.values if line[-1] == -1 ])
    ones = np.array(sample([ line for line in data.values if line[-1] == 1 ], len(minus_ones)))
    zeros = np.array(sample([ line for line in data.values if line[-1] == 0 ], len(minus_ones)))
    data = np.vstack([minus_ones, ones, zeros])

    # Make feature matrix
    y_data = data[:,-1]
    X_data = data[:,:-1]

    tfidf_comment = TfidfVectorizer(ngram_range=(1,2), max_features=150)
    comment_sparse = tfidf_comment.fit_transform(X_data[:,0])
    scaler = StandardScaler()
    sentim = scaler.fit_transform(X_data[:,2:4])
    sarc = [ [elem] for elem in X_data[:,3] ]
    X_sparse = hstack([comment_sparse, sentim, sarc])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y_data, test_size=0.2)

    return X_train, X_test, y_train, y_test, tfidf_comment, scaler


def encode_labels(y_train, y_test):
    # Use One Hot Encoding to encode categories
    y_train_encoded = to_categorical(y_train, 3)
    y_test_encoded = to_categorical(y_test, 3)

    return y_train_encoded, y_test_encoded


def build_classifier(X_train, y_train, X_test, y_test):
    # Create the model
    model = Sequential()
    model.add(Dense(20, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(3, activation="softmax")) 

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

    return model, history


def test_classifier(model, X_test, y_test, history, showMode=False):
    # Plot accuracy
    def plot_history(history, key='accuracy'):
        plt.figure(figsize=(16,10))

        val = plt.plot(history.epoch, history.history['val_'+key], '--', label='Accuracy on validation set')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=' Accuracy on training set')

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_',' ').title())
        plt.legend()

        plt.xlim([0,max(history.epoch)])

        if showMode:
            plt.show() 
        else:
            plt.savefig('img/neural_network_training.png')

    plot_history(history)

    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)

    # Determine category from One Hot Encoding
    y_pred = categorize(y_pred)
    y_test = categorize(y_test)

    if showMode:
        print('=============================')
        print(y_pred)
        print(y_test)
        print('=============================')

    # Plot confusion matrix
    cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
    print(cf_matrix)
    heatmap(cf_matrix, cmap='Blues')
    if showMode:
        plt.show()
    else:
        plt.savefig('img/neural_network_confusion_matrix')

    # Print metrics
    for i, metric in enumerate(metrics):
        print(model.metrics_names[i], '=', metric)
 

def categorize(encoded):
    a = [ argmax(line) for line in encoded ]
    for i, l in enumerate(a):
        a[i] = l if l <= 1 else -1
    return a


def test_classifier_web(clf, tfidf_comment, scaler, X_data):
    print(X_data)
    comment_sparse = tfidf_comment.transform(X_data['body'])
    sentim = scaler.transform(X_data.iloc[:,2:4])
    sarc = [ [elem] for elem in X_data.iloc[:,3] ]

    print(comment_sparse.shape, sentim.shape, sarc)
    X_sparse = hstack([comment_sparse, sentim, sarc])
    X_sparse = X_sparse.toarray()
    print(X_sparse)
    y_pred = clf.predict(X_sparse)

    return categorize(y_pred)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    y_train, y_test = encode_labels(y_train, y_test)
    model, history = build_classifier(X_train, y_train, X_test, y_test)
    test_classifier(model, X_test, y_test, history)
