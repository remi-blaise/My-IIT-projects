# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""

# add whatever imports you need.
# be sure to also add to requirements.txt so I can install them.
import click
import json
import glob
import pickle
import sys
import numpy as np
import os
import pandas as pd
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report


from . import NB_path, NN_path, LR_path, Sarcasm_path
from subprocess import call

from .data_collection.collect_comments import collect_comments, politifact_claims
from .stats.analyse_data import read_train_comments, analyse_data
from .classifiers.naive_bayes import get_data, NaiveBayes, open_doc, compute_stats
from .classifiers.neural_network import get_data as neural_get_data, encode_labels, build_classifier, test_classifier
from .classifiers.LogReg import train_clf, test_clf
from .data_collection.sarcasm.detect_sarcasm import detect_sarcasm


@click.group()
def main(args=None):
    """Console script for osna."""
    return 0


@main.command('collect')
def collect():
    """
    Collect data and store in given directory.

    This should collect any data needed to train and evaluate your approach.
    This may be a long-running job (e.g., maybe you run this command and let it go for a week).
    """
    collect_comments(politifact_claims())
    path = 'osna/data_collection/'
    call(["python3", path + "shuffle.py"])


@main.command('evaluate')
def evaluate():
    """
    Report accuracy and other metrics of your approach.
    For example, compare classification accuracy for different
    methods.
    """
    # Naïve Bayes
    NB, test_data = pickle.load(open(NB_path, 'rb'))
    text_ids, gold_labels, pred_labels, pred_probs = NB.predict(test_data)
    compute_stats(gold_labels, pred_labels, showMode=True)
    
    # Logistic Regression
    clf,tfidf_comment,scaler,scaler2,X_test,y_test = pickle.load(open(LR_path, 'rb'))
    test_clf(clf,tfidf_comment,scaler,scaler2,X_test,y_test, showMode=True)
    
    # Neural network
    model, X_test, y_test, history, tfidf_comment, scaler = pickle.load(open(NN_path, 'rb'))
    test_classifier(model, X_test, y_test, history, showMode=True)


@main.command('network')
def network():
    """
    Perform the network analysis component of your project.
    E.g., compute network statistics, perform clustering
    or link prediction, etc.
    """
    print('We did not see any relevant network analysis to make on this data.')


@main.command('stats')
def stats():
    """
    Read all data and print statistics.
    E.g., how many messages/users, time range, number of terms/tokens, etc.
    """
    #print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.
    analyse_data(read_train_comments(), showMode=True)


@main.command('train')
#@click.argument('directory', type=click.Path(exists=True))
def train():
    """
    Train a classifier on all of your labeled data and save it for later
    use in the web app. You should use the pickle library to read/write
    Python objects to files. You should also reference the `clf_path`
    variable, defined in __init__.py, to locate the file.
    """
    # Sentiment analysis
    path_SA = './osna/sentiment_analysis/'
    call(["python3", path_SA + "analyse_sentiment_naive_bayes.py"])
    call(["python3", path_SA + "analyse_sentiment_usingtextblob.py"])

    # # Sarcasm
    tfidf_comment, clf_sarcasm= detect_sarcasm(showMode=False)
    pickle.dump((tfidf_comment, clf_sarcasm), open(Sarcasm_path, 'wb'))
    
    # Naïve Bayes
    print('Training with Naive Bayes')
    threshold = 0.8
    table = open_doc("./osna/data_collection/commentssarc.csv",';')
    belief_comments, nonbelief_comments, train_belief, train_nonbelief, test_data = get_data(table, threshold)
    NB = NaiveBayes(belief_comments, nonbelief_comments, train_belief, train_nonbelief) 
    pickle.dump((NB, test_data), open(NB_path, 'wb'))
    
    # Logistic Regression
    print('Training with Logistic Regression')
    clf,tfidf_comment,scaler,scaler2,X_test,y_test = train_clf()
    pickle.dump((clf,tfidf_comment,scaler,scaler2,X_test,y_test), open(LR_path, 'wb'))
    
    # Neural network
    print('Training with Neural network')
    X_train, X_test, y_train, y_test, NN_tfidf_comment, NN_scaler = neural_get_data()
    y_train, y_test = encode_labels(y_train, y_test)
    model, history = build_classifier(X_train, y_train, X_test, y_test)
    pickle.dump((model, X_test, y_test, history, NN_tfidf_comment, NN_scaler), open(NN_path, 'wb'))


@main.command('web')
@click.option('-p', '--port', required=False, default=9999, show_default=True, help='port of web server')
def web(port):
    """
    Launch a web app for your project demo.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
