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
from itertools import chain
import seaborn as sns
import matplotlib.pyplot as plt

ALPHA = 1.0

def get_data(table, threshold):
    '''Get necessary data for classifier'''
    belief_comments, nonbelief_comments, none_comments = divideByLabel(table[['body', 'label']])
    train_belief, test_belief = shuffleAndDivideTrainTest(belief_comments, threshold, len(nonbelief_comments))
    train_nonbelief, test_nonbelief = shuffleAndDivideTrainTest(nonbelief_comments, threshold, len(nonbelief_comments))
    test_data = test_belief + test_nonbelief
    return belief_comments, nonbelief_comments, train_belief, train_nonbelief, test_data

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
    
def divideByLabel(table):
    ''' Classify comments into 3 classes:
        Belief with label 1
        Non-belief with label -1
        None of the above with label 0
    '''
    belief_comments = dict()
    nonbelief_comments = dict()
    none_comments = dict()

    for i, comment in enumerate(table['body']):
        if table.iloc[i]['label'] == 1:
            belief_comments[i] = comment
        if table.iloc[i]['label'] == -1:
            nonbelief_comments[i] = comment
        if table.iloc[i]['label'] == 0:
            none_comments[i] = comment
    return belief_comments, nonbelief_comments, none_comments

def shuffleAndDivideTrainTest(comments, threshold, size):
    '''Divide shuffled comments into train and test data'''
    commentsAsList = list(comments.items())
    shuffle(commentsAsList)
    if size != 0:
        #Have both data class at same size
        commentsAsList = commentsAsList[0:size]
    threshold_id = trunc(len(commentsAsList)*threshold)
    train = commentsAsList[0:threshold_id]
    test = commentsAsList[threshold_id:len(commentsAsList)]
    return train, test
    
def cross_validation_accuracy(fold):
    '''Classify comments with Naïve Bayes multiple times (fold)
        Compute accuracy each time
    '''
    accuracies = []
    threshold = 0.8
    print('\nlenght of each class pos: ', len(belief_comments), ',  class neg: ', len(nonbelief_comments))
    for i in range(0, fold):
        #Generate train and test data 
        belief_comments, nonbelief_comments, train_belief, train_nonbelief, test_data = get_data(table, threshold)
        
        #Do naïve bayes classification
        NB = NaiveBayes(belief_comments, nonbelief_comments, train_belief, train_nonbelief)
        text_ids, gold_labels, pred_labels, pred_probs = NB.predict(test_data)
        
        #Compute accuracy
        accuracies.append(np.sum(np.equal(pred_labels, gold_labels)) / float(len(gold_labels)))
    
    #Keep last fold result in csv
    # pandas.DataFrame({"File ID": text_ids,
                  # "Class": gold_labels,
                  # "Predicted Class": pred_labels,
                  # "Predicted Probability": pred_probs}).to_csv("predictions.csv")
                  
    return accuracies, gold_labels, pred_labels
    
def analyse_model(gold_labels, pred_labels, gold_class):
    '''Compute confusion matrix and analyze model for a given class (gold_class)'''
    confusion_matrix = defaultdict(int)
    measures = defaultdict(int)
    
    for gold_label, pred_label in zip(gold_labels, pred_labels):
        # true positives: items IN given class that are predicted to be IN that class
        if gold_label == gold_class and pred_label == gold_class:
            confusion_matrix['true_pos'] += 1
        
        # false positives: items NOT in given class that are predicted to be IN that class
        if gold_label == -gold_class and pred_label == gold_class:
            confusion_matrix['false_pos'] += 1
        
        # true negatives: items NOT in given class that are NOT predicted to be in that class
        if gold_label == -gold_class and pred_label == -gold_class:
            confusion_matrix['true_neg'] += 1
            
        # false negatives: items IN given class that are NOT predicted to be in that class
        if gold_label == gold_class and pred_label == -gold_class:
            confusion_matrix['false_neg'] += 1
    
    if confusion_matrix['true_pos'] != 0:
        measures['precision'] = round(confusion_matrix['true_pos'] / (confusion_matrix['true_pos'] + confusion_matrix['false_pos']), 3)
        measures['recall'] = round(confusion_matrix['true_pos'] / (confusion_matrix['true_pos'] + confusion_matrix['false_neg']), 3)
        measures['F_measure'] = round( (2 * measures['precision'] * measures['recall']) / (measures['precision'] + measures['recall']) , 3)
    
    return confusion_matrix, measures
    
def compute_stats(gold_labels, pred_labels, showMode=False):
    '''Compute statistics for both classes'''
    #One is just inverse of the other
    confusion_matrix_pos, measures_pos = analyse_model(gold_labels, pred_labels, 1)
    confusion_matrix_neg, measures_neg = analyse_model(gold_labels, pred_labels, -1)
    if showMode:
        print('Positive class: ', sorted(confusion_matrix_pos.items(), key=lambda x: x[0]) + list(measures_pos.items()))
        print('Negative class: ', sorted(confusion_matrix_neg.items(), key=lambda x: x[0]) + list(measures_neg.items()))
    
    cf_matrix = [[0, 0], [0, 0]]
    # well-classfied
    cf_matrix[0][0] = confusion_matrix_pos['true_pos']
    cf_matrix[1][1] = confusion_matrix_pos['true_neg']
    # wrongly-classified
    cf_matrix[0][1] = confusion_matrix_pos['false_neg']
    cf_matrix[1][0] = confusion_matrix_pos['false_pos']
    sns.heatmap(cf_matrix, cmap='Blues')
    if showMode:
        plt.show()
    else:
        plt.savefig('img/NB_confusion_matrix.png')

    return measures_pos, measures_neg
    
class NaiveBayes:
    '''Naive Bayes text categorization model
        Inspired from NLP class
    '''

    def __init__(self, belief_comments, nonbelief_comments, train_belief, train_nonbelief):
        # To keep in object for evaluation in osna commands
        self.nonbelief_comments = nonbelief_comments
        self.belief_comments = belief_comments
        
        self.train(train_belief, train_nonbelief)

    def train(self, train_belief, train_nonbelief):
        '''Train model by collecting counts from training data'''
        # Counts of words in belief-/nonbelief-class texts
        self.count_belief = Counter()
        self.count_nonbelief = Counter()

        # Total number of comments for each category
        self.num_belief_comments = 0
        self.num_nonbelief_comments = 0

        # Total number of words in positive/negative comments
        self.total_belief_words = 0
        self.total_nonbelief_words = 0

        # Class priors (logprobs)
        # log(P(y=pos))
        self.p_belief = 0.0
        # log(P(y=neg))
        self.p_nonbelief = 0.0

        # Iterate through texts and collect count statistics initialized above
        for i, comment in train_belief:
            self.count_belief.update(tokenize(comment))
            self.num_belief_comments += 1
            self.total_belief_words += len(list(tokenize(comment)))

        for i, comment in train_nonbelief:
            self.count_nonbelief.update(tokenize(comment))
            self.num_nonbelief_comments += 1
            self.total_nonbelief_words += len(list(tokenize(comment)))

        # Calculate derived statistics
        self.vocab = set(list(self.count_nonbelief.keys())
                         + list(self.count_belief.keys()))
        self.p_belief = log(float(self.num_belief_comments)) \
            - log(float(self.num_belief_comments + self.num_nonbelief_comments))
        self.p_nonbelief = log(float(self.num_nonbelief_comments)) \
            - log(float(self.num_belief_comments + self.num_nonbelief_comments))

    def predict(self, test_data):
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
        
        for i, comment in test_data:
            print(comment)
            comment_ids.append(i)
            if comment == self.belief_comments.get(i):
                gold_labels.append(1.0)
            elif comment == self.nonbelief_comments.get(i):
                gold_labels.append(-1.0)

            # Implement naive Bayes probability estimation to calculate class probabilities
            # and predicted labels for each text in the test set
            
            # log(P(X|Y=pos))
            logP_XgivenYpos = 0
            # log(P(X|Y=neg))
            logP_XgivenYneg = 0
            
            for j, word in enumerate(tokenize(comment)):
                word = word.lower()
                logP_XgivenYpos += (log(float(self.count_belief[word] + ALPHA)) - log(float(self.total_belief_words + ALPHA*len(self.vocab))) )
                logP_XgivenYneg += (log(float(self.count_nonbelief[word] + ALPHA)) - log(float(self.total_nonbelief_words + ALPHA*len(self.vocab))) )
        
            # log(P(Pos,X))
            sum_belief = logP_XgivenYpos + self.p_belief
            # log(P(Neg,X))
            sum_nonbelief = logP_XgivenYneg + self.p_nonbelief      

            # Get P(Y|X) by normalizing across log(P(Y,X)) for both values of Y
            # 1) Get K = log(P(Pos|X) + P(Neg|X))
            normalization_factor = self.log_sum(sum_belief, sum_nonbelief)
            # 2) Calculate P(Pos|X) = e^(log(P(Pos,X)) - K)
            predicted_prob_positive = exp(sum_belief - normalization_factor)
            # 3) Get P(Neg|X) = P(Neg|X) = e^(log(P(Neg,X)) - K)
            predicted_prob_negative = 1.0 - predicted_prob_positive

            pred_probs.append(predicted_prob_positive)
            if predicted_prob_positive > predicted_prob_negative:
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)

        print(pred_labels)
        return comment_ids, gold_labels, pred_labels, pred_probs
        
    def log_sum(self, logx, logy):
        '''Utility function to compute $log(exp(logx) + exp(logy))$
        while avoiding numerical issues
        '''
        m = max(logx, logy)
        return m + log(exp(logx - m) + exp(logy - m))
    
if __name__ == "__main__":
    table=open_doc("./osna/data_collection/commentssarc.csv",';')

    #Divide comments into 3 classes
    belief_comments, nonbelief_comments, none_comments = divideByLabel(table[['body', 'label']])
    print('\nlenght of each class pos: ', len(belief_comments), ',  class neg: ', len(nonbelief_comments), ',  class neut: ', len(none_comments))
    
    
    #Naïves Bayes classification with cross validation
    accuracies, gold_labels, pred_labels = cross_validation_accuracy(10)
    print ("alpha = ", ALPHA)
    print("Mean accuracy: {:.2f}%".format(100 * np.mean(accuracies)))
    #pandas.DataFrame({"Accuracy": accuracies}).to_csv("NB_accuracies.csv")
    
    
    
    
    
