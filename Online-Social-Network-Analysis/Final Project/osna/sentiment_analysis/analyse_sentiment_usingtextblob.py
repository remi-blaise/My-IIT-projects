#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas
from textblob import TextBlob

DATA_PATH = './osna/data_collection/'

if __name__ == "__main__":
    # get comments
    table = pandas.read_csv(DATA_PATH + 'commentssarc.csv', sep=';')
    bodies = table['body']

    # classify with Textblob and put result in a csv file
    sentiments = [ TextBlob(body).sentiment for body in bodies ]
    pandas.DataFrame({
        'polarity': [ sentiment.polarity for sentiment in sentiments ],
        'subjectivity': [ sentiment.subjectivity for sentiment in sentiments ],
    }).to_csv('./osna/sentiment_analysis/sentiments_usingtextblob.csv')
