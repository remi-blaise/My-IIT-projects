# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:48:19 2019

@author: alanc
"""

import pandas as pd
import numpy as np
import pickle

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



def open_doc(path,sepa):
    table = pd.read_csv(path, sep=sepa)
    return table

def train_clf():
    reader=open_doc("./osna/data_collection/commentssarc.csv",';')
    sentiments=open_doc("./osna/sentiment_analysis/sentiments_usingtextblob.csv",',')

	#Get paths of features
    reader['polarity']=sentiments['polarity']
    reader['subjectivity']=sentiments['subjectivity']
    reader['len']=[len(i) for i in reader['body']]

    features=['polarity','subjectivity','body','sarcarsm','len','label']
    data=reader[features]
    y_data=data['label']
	#y_data=(data['label']==1)*1
    X_data=data.drop(['label'],axis=1)


	#Data prep (Tfidf and Scaling)
    tfidf_comment = TfidfVectorizer(ngram_range=(1,2), max_features=150)
    comment_sparse = tfidf_comment.fit_transform(X_data['body'])
    scaler = StandardScaler()
    sentim = scaler.fit_transform(X_data.iloc[:,:2])
    sarc=[[elem] for elem in X_data['sarcarsm'].values]
    lenl=[[elem] for elem in X_data['len'].values]
    scaler2 = StandardScaler()
    lenl= scaler2.fit_transform(lenl)

	#Make a single parse matrix
    X_sparse = hstack([comment_sparse,sarc,sentim,lenl])
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y_data,test_size=0.3)
	#Give weigth to classes
    weight_class={-1:14.5,0:1.5,1:1}
    clf = LogisticRegression(solver='lbfgs', verbose=False, max_iter=10000, class_weight=weight_class)
    clf.fit(X_train, y_train)

    #Wordloud for class 1 (Trust)
    sincere_comments = str(data[data['label'] == 1]['body'])
    plt.figure(figsize=(12, 12))
    word_cloud = WordCloud(stopwords=STOPWORDS)
    word_cloud.generate(sincere_comments)
    plt.imshow(word_cloud)
    plt.title('Label=1')
#    if showMode:
#        plt.imshow(word_cloud)
#    else:
#        plt.savefig('./img/LR_Workcloud1')

    #Wordloud for class 0 (Neutral)
    sincere_comments = str(data[data['label'] == 0]['body'])
    plt.figure(figsize=(12, 12))
    word_cloud = WordCloud(stopwords=STOPWORDS)
    plt.title('Label=0')
    word_cloud.generate(sincere_comments)
    plt.imshow(word_cloud)
#    if showMode:
#        plt.imshow(word_cloud)
#    else:
#        plt.savefig('./img/LR_Workcloud0')

    #Wordloud for class -1 (No Trust)
    sincere_comments = str(data[data['label'] == -1]['body'])
    plt.figure(figsize=(12, 12))
    plt.title('Label=-1')
    word_cloud = WordCloud(stopwords=STOPWORDS)
    word_cloud.generate(sincere_comments)
    plt.imshow(word_cloud)
    print("Testing accuracy",clf.score(X_train, y_train))
    return clf,tfidf_comment,scaler,scaler2,X_test,y_test


def test_clf(clf,tfidf_comment,scaler,scaler2,X_test,y_test, showMode=True):
    y_pred=clf.predict(X_test)

	#Tests of accuracy/ Confusion matrix
#    if showMode:
#        print("Testing accuracy",clf.score(X_test, y_test))
    plt.figure()
    cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
    sumcf=np.sum(cf_matrix,axis=1)
    for i in range(3):
        cf_matrix[i]=[float(elem) for elem in cf_matrix[i]]
        cf_matrix[i] = cf_matrix[i]*100/float(sumcf[i])

    sns.heatmap(cf_matrix, cmap='Blues')
    plt.title('Confusion matrix')


def test_clf_web(clf,tfidf_comment,scaler,scaler2,X_data, showMode=True):
    comment_sparse = tfidf_comment.transform(X_data['body'])
    sentim = scaler.transform(X_data.iloc[:,:2])
    X_data['len']=[len(i) for i in X_data['body']]
    sarc=[[elem] for elem in X_data['sarcarsm'].values]
    lenl=[[elem] for elem in X_data['len'].values]
    lenl= scaler2.transform(lenl)
    X_sparse = hstack([comment_sparse,sarc,sentim,lenl])
    y_pred=clf.predict(X_sparse)
#    X_data['predicted']=y_pred
    return(y_pred)
