from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
#from ..mytwitter import Twitter
#from ..u import get_twitter_data, N_TWEETS

import pickle
import sys

from .. import NB_path, NN_path, LR_path, Sarcasm_path

from ..data_collection.collect_comments import collect_comments
from ..stats.analyse_data import analyse_data
from ..classifiers.naive_bayes import get_data, NaiveBayes, open_doc, compute_stats
from ..classifiers.neural_network import get_data as neural_get_data, encode_labels, build_classifier, test_classifier, test_classifier_web
from ..classifiers.LogReg import train_clf, test_clf, test_clf_web
from textblob import TextBlob

import time
import pandas

DATA_PATH = './osna/data_collection/'
STAT_PATH = './osna/stats/'

#twapi = Twitter(credentials_path)
# clf, vec = pickle.load(open(clf_path, 'rb'))
# print('read clf %s' % str(clf))
# print('read vec %s' % str(vec))

@app.route('/', methods=['GET'])
def index():
    form = MyForm()
    return render_template('myform.html', form=form)

@app.route('/result', methods=['POST'])
def result():
    form = MyForm()
    claim = form.input_field.data

    # Collect comments from Reddit
    collect_comments([(0, claim)], output_filename='comments_web.csv')
    comments = pandas.read_csv(DATA_PATH + 'comments_web.csv')

    # Predict sarcasm
    tfidf_comment, clf_sarcasm = pickle.load(open(Sarcasm_path, 'rb'))
    sarcasm_comments = tfidf_comment.transform(comments['body'])
    sarcasm = clf_sarcasm.predict(sarcasm_comments)
    comments['sarcarsm'] = sarcasm

    # Predict sentiment
    sentiments = [ TextBlob(body).sentiment for body in comments['body'] ]
    comments['polarity'] = [ sentiment.polarity for sentiment in sentiments ]
    comments['subjectivity'] = [ sentiment.subjectivity for sentiment in sentiments ]

    # Predict labels
    # Naïve Bayes
    NB, test_data = pickle.load(open(NB_path, 'rb'))
    # text_ids, gold_labels, pred_labels, pred_probs = NB.predict(test_data)
    # NB_measures_pos, NB_measures_neg = compute_stats(gold_labels, pred_labels, showMode=False)
    _, _, NB_pred_labels, NB_pred_probs = NB.predict(enumerate(comments['body']))
    print(_, _, NB_pred_labels, NB_pred_probs)
    
    # Logistic Regression
    features=['polarity','subjectivity','body','sarcarsm']
    data=comments[features]
    LR_clf, LR_tfidf_comment, LR_scaler, LR_scaler2, LR_X_test, LR_y_test = pickle.load(open(LR_path, 'rb'))
    LR_pred_labels = test_clf_web(LR_clf, LR_tfidf_comment, LR_scaler, LR_scaler2, data, showMode=False)
    
    # # Neural network
    features = ['body', 'polarity', 'subjectivity', 'sarcarsm']
    data = comments[features]
    model, X_test, y_test, history, NN_tfidf_comment, NN_scaler = pickle.load(open(NN_path, 'rb'))
    NN_pred_labels = test_classifier_web(model, NN_tfidf_comment, NN_scaler, data)

    # Get metrics
    # Naïve Bayes
    NB_comments = comments.copy()
    # NB_comments.info()
    # print(NB_pred_labels)
    NB_comments['label'] = NB_pred_labels
    NB_labels, NB_articles, NB_authors, NB_count_tokens = analyse_data(NB_comments, showMode=False, img_prefix='../app/static/NB_')
    len_NB_articles, len_NB_authors, len_NB_count_tokens = len(NB_articles), len(NB_authors), len(NB_count_tokens)

    # Format results
    NB_pred_labels = list(map(int, NB_pred_labels))
    LR_pred_labels = list(map(int, LR_pred_labels))

    result = [ (comment, NB_pred_labels[i], LR_pred_labels[i], NN_pred_labels[i]) for i, comment in enumerate(comments['body']) ]

    return render_template('result.html', 
        claim=claim, 
        result=result,
        len_NB_articles=len_NB_articles, len_NB_authors=len_NB_authors, len_NB_count_tokens=len_NB_count_tokens, NB_labels=NB_labels,
        nb_comments=len(comments),
        timestamp=time.time()
    )
