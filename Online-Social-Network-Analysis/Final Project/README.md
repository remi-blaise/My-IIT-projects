# Requirements

The following database is necessary to run the Sarcasm detection (too heavy for GitHub) : https://www.kaggle.com/danofer/sarcasm
Tensorflow does not work if it is in requirements. It is necessary to download it independently.

# Do people believe fake news?

This is a project for CS579. See [Instructions.md](Instructions.md) to get started.
This file contain a high-level summary of our project. It is a condensed version of our report.

## Introduction

The goal of the project is to make a classiﬁer able to guess if a user believes or not to a news article via one of his comments. 
Applications of a such classiﬁer could be: analysing fake news propagation; analysing people’s credulity and critical thinking about these news

## Data

We found a database of fake news from Politifact, Snopes and Emergent websites found the Kaggle website. Every ﬁrst-comments (no responses to comments) associated to each news post on Reddit were then collected. 
Among the 13000 comments gathered from Politifact’s news' headlines, we labelized 3000 random comments with the labels: -1 for non-believer; 1 for believer; 0 for non-belief related or ambigious comments.

## Methods

In order to get features, we used a bag of words representation, sentiment analysis by an existing model and a lexicon based method, and a sarcasm indicator provided by an existing model. 
We made three classifiers with supervised learning algorithms: naive Bayes, logistic regression and simple neural networks.
Programming tools such as pandas for data handling, numpy, sklearn for machine learning tasks, keras for neural network tasks and textblob for NLP were used for this project.

## Results

For category "non-believer", we have 204 comments
For category "believer", we have 1992 comments 
For category "none of the above", we have 798 comments
We have 1831 auhtors in our labelized data and most users comment only once in our database. The most active one has written 30 comments.

On a test database, we have for 3 classes: logistic regression accuracy is 49%, simple neural network accuracy is 46% and for 2 classes: naive Bayes accuracy is 56%
