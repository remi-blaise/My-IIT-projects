from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from collections import Counter
from math import log, exp, trunc
import re
import sys
import numpy as np
import pandas
from random import shuffle

ALPHA = 1.0

def open_doc(path,sepa):
    '''Get comments'''
    table = pandas.read_csv(path, sep=sepa)
    return table

def tokenize(doc, keep_internal_punct=True):
    '''Tokenize a sentence into words'''
    if not doc:
        return []
    
    tokens = []
    doc = doc.lower()
    if keep_internal_punct:
        for word in doc.split():
            word = re.sub(r"^\W+", "", word)
            word = re.sub(r"\W+$", "", word)
            tokens.append(word)     
    else:
        tokens = re.sub('\W+', ' ', doc).split()
    return tokens
        
def get_lexicon():
    '''Get Afinn lexicon'''
    afinn = dict()
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    return afinn

def afinn_sentiment(terms, afinn, verbose=False):
    '''Compute sentiment score based on Afinn lexicon'''
    score = 0.
    for t in terms:
        if t in afinn:
            score += afinn[t]
    return score
    
def classify_lexicon_based(comments):
    ''' Classify comments into 2 classes:
        Positive with sentiment score > 0
        Negative with sentiment score <= 0
        
        Put neutral in negative because 0 strictly negative comments in labelized data
    '''
    positive_comments = dict()
    negative_comments = dict()
    
    for i, comment in enumerate(comments):
        tokens = tokenize(comment)
        sentiment_score = afinn_sentiment(tokens, afinn)
        if sentiment_score > 0:
            positive_comments[i] = comment
        if sentiment_score <= 0:
            negative_comments[i] = comment
    return positive_comments, negative_comments

def shuffleAndDivideTrainTest(comments, threshold=0.9):
    '''Divide shuffled comments into train and test data'''
    commentsAsList = list(comments.items())
    shuffle(commentsAsList)
    threshold_id = trunc(len(comments)*threshold)
    train = commentsAsList[0:threshold_id]
    test = commentsAsList[threshold_id:len(comments)]
    return train, test
    
def cross_validation_accuracy(fold):
    '''Classify comments with Naïve Bayes multiple times (fold)
        Compute accuracy each time
    '''
    accuracies = []
    threshold = 0.9
    for i in range(0, fold):
        #Generate train and test data 
        train_pos_comments, test_pos_comments = shuffleAndDivideTrainTest(positive_comments, threshold)
        train_neg_comments, test_neg_comments = shuffleAndDivideTrainTest(negative_comments, threshold)
        test_data = test_pos_comments + test_neg_comments
        
        #Do naïve bayes classification
        NB = NaiveBayes(train_pos_comments, train_neg_comments)
        test_ids, gold_labels, pred_labels, pred_probs = NB.predict(test_data)
        
        #Compute accuracy
        accuracies.append(np.sum(np.equal(pred_labels, gold_labels)) / float(len(gold_labels)))
    
    #Keep last fold result in csv
    pandas.DataFrame({"File ID": test_ids,
                  "Class": gold_labels,
                  "Predicted Class": pred_labels,
                  "Predicted Probability": pred_probs}).to_csv("./osna/sentiment_analysis/predictions.csv")
                  
    return accuracies
    
class NaiveBayes:
    '''Naive Bayes text categorization model
        Inspired from NLP class
    '''

    def __init__(self, train_pos_comments, train_neg_comments):
        self.train(train_pos_comments, train_neg_comments)

    def train(self, train_pos_comments, train_neg_comments):
        '''Train model by collecting counts from training data'''
        # Counts of words in positive-/negative-class texts
        self.count_positive = Counter()
        self.count_negative = Counter()

        # Total number of comments for each category
        self.num_positive_comments = 0
        self.num_negative_comments = 0

        # Total number of words in positive/negative comments
        self.total_positive_words = 0
        self.total_negative_words = 0

        # Class priors (logprobs)
        # log(P(y=pos))
        self.p_positive = 0.0
        # log(P(y=neg))
        self.p_negative = 0.0

        # Iterate through texts and collect count statistics initialized above
        for i, comment in train_pos_comments:
            self.count_positive.update(tokenize(comment))
            self.num_positive_comments += 1
            self.total_positive_words += len(list(tokenize(comment)))

        for i, comment in train_neg_comments:
            self.count_negative.update(tokenize(comment))
            self.num_negative_comments += 1
            self.total_negative_words += len(list(tokenize(comment)))

        # Calculate derived statistics
        self.vocab = set(list(self.count_negative.keys())
                         + list(self.count_positive.keys()))
        self.p_positive = log(float(self.num_positive_comments)) \
            - log(float(self.num_positive_comments + self.num_negative_comments))
        self.p_negative = log(float(self.num_negative_comments)) \
            - log(float(self.num_positive_comments + self.num_negative_comments))

    def predict(self, data):
        """For each comment
           - append the comment id (file basename) to `comment_ids`
           - append the predicted label (1.0 or -1.0) to `pred_labels`
           - append the correct (gold) label (1.0 or -1.0) to `gold_labels`
           - append the probability of the positive (1.0) class to `pred_probs`
        """
        comment_ids = []
        pred_labels = []
        pred_probs = []
        gold_labels = []
        
        for i, comment in data:
            comment_ids.append(i)
            if comment == positive_comments.get(i):
                gold_labels.append(1.0)
            elif comment == negative_comments.get(i):
                gold_labels.append(-1.0)

            # Implement naive Bayes probability estimation to calculate class probabilities
            # and predicted labels for each text in the test set
            
            # log(P(X|Y=pos))
            logP_XgivenYpos = 0
            # log(P(X|Y=neg))
            logP_XgivenYneg = 0
            
            for j, word in enumerate(tokenize(comment)):
                word = word.lower()
                logP_XgivenYpos += (log(float(self.count_positive[word] + ALPHA)) - log(float(self.total_positive_words + ALPHA*len(self.vocab))) )
                logP_XgivenYneg += (log(float(self.count_negative[word] + ALPHA)) - log(float(self.total_negative_words + ALPHA*len(self.vocab))) )
        
            # log(P(Pos,X))
            sum_positive = logP_XgivenYpos + self.p_positive
            # log(P(Neg,X))
            sum_negative = logP_XgivenYneg + self.p_negative        

            # Get P(Y|X) by normalizing across log(P(Y,X)) for both values of Y
            # 1) Get K = log(P(Pos|X) + P(Neg|X))
            normalization_factor = self.log_sum(sum_positive, sum_negative)
            # 2) Calculate P(Pos|X) = e^(log(P(Pos,X)) - K)
            predicted_prob_positive = exp(sum_positive - normalization_factor)
            # 3) Get P(Neg|X) = P(Neg|X) = e^(log(P(Neg,X)) - K)
            predicted_prob_negative = 1.0 - predicted_prob_positive

            pred_probs.append(predicted_prob_positive)
            if predicted_prob_positive > predicted_prob_negative:
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)

        return comment_ids, gold_labels, pred_labels, pred_probs
        
    def log_sum(self, logx, logy):
        '''Utility function to compute $log(exp(logx) + exp(logy))$
        while avoiding numerical issues
        '''
        m = max(logx, logy)
        return m + log(exp(logx - m) + exp(logy - m))
    
if __name__ == "__main__":
    table=open_doc("./osna/data_collection/commentssarc.csv",';')
    afinn = get_lexicon()

    #Lexicon-based classification using Afinn as lexicon
    positive_comments, negative_comments = classify_lexicon_based(table['body'])
    #print('\nlenght of each class pos: ', len(positive_comments), ',  class neg: ', len(negative_comments))
    
    #Naïves Bayes classification with cross validation
    accuracies = cross_validation_accuracy(10)
    #print ("alpha = ", ALPHA)
    #print("Mean accuracy: {:.2f}%".format(100 * np.mean(accuracies)))
    
    
    
