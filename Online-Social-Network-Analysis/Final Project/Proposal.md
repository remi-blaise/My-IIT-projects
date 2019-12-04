## Overview

Do people believe fake news?

The goal is to make a classifier able to guess if a user believes or not to a news article via one of his comment. It will involve NLP methods similar to sentiment analysis. It can then be used to analyse fake news propagation ; people's credulity and critical thinking about these news ; how much is it easy to manipulate people ; are there group movements and changes of mid over time and what influence can popular people have.

## Data

We will try to get data from social networks like Reddit and Twitter (through APIs) and maybe from other sources like Kaggle.



A big problem we anticipate is how hard it will be to get labelled data to train the classifier. Indeed, most of works on the topic is related to fake news detection and sentiment analysis.

## Method

Usual machine learning algorithms, neural networks and text analysis. We foresee to use librairies like scykit, tensorflow/keras, pandas, textblob

## Related Work

https://paperswithcode.com/paper/fake-news-detection-on-social-media-a-data

https://paperswithcode.com/paper/recurrent-attention-network-on-memory-for

https://paperswithcode.com/paper/defending-against-neural-fake-news

https://paperswithcode.com/paper/learning-to-generate-reviews-and-discovering

https://paperswithcode.com/paper/using-millions-of-emoji-occurrences-to-learn

## Evaluation

By separating training/testing sets, using cross-validation method to have more reliability and manually checking a portion of the result.



Graphs: given a news article

- evolution of the number of believers/non believers over time 

- number of tweets on the topic on a timeline 

- influence of most popular people by comparing beliefs of their community with general credence 

