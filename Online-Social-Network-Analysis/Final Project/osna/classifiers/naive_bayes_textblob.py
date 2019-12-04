import pandas
from textblob.classifiers import NaiveBayesClassifier
from random import shuffle
from math import trunc
import numpy as np

def open_doc(path,sepa):
    table = pandas.read_csv(path, sep=sepa)
    return table
	
def formatData(table):
	'''Format data for Textblob classifier'''
	commentsLabelled = []
	for i, comment in enumerate(table['body']):
		if table.iloc[i]['label'] == 1.0:
			commentsLabelled.append((comment, 'pos'))
		if table.iloc[i]['label'] == -1.0:
			commentsLabelled.append((comment, 'neg'))
	return commentsLabelled
	
def shuffleAndDivideTrainTest(comments, threshold=0.9):
	'''Divide shuffled comments into train and test data'''
	shuffle(comments)
	threshold_id = trunc(len(comments)*threshold)
	train = comments[0:threshold_id]
	test = comments[threshold_id:len(comments)]
	return train, test
	
def predict(test):
	'''Predict labels on test data'''
	i = 0
	ids = []
	pred_labels = []
	gold_labels = []
	pos_pred_probs = []
	
	for comment, label in test:
		prob_dist = cl.prob_classify(comment)
		ids.append(i)
		gold_labels.append(label)
		pred_labels.append(prob_dist.max())
		pos_pred_probs.append(round(prob_dist.prob("pos"), 2))
		i+=1
	
	return ids, gold_labels, pred_labels, pos_pred_probs
	
if __name__ == "__main__":
	table=open_doc("./osna/data_collection/commentssarc.csv",';')
	commentsLabelled = formatData(table)
	
	accuracies = []
	#Numerous folds take lots of memory, could not do more
	for i in range(0, 1):
		train, test = shuffleAndDivideTrainTest(commentsLabelled)
		cl = NaiveBayesClassifier(train)
		accuracies.append(cl.accuracy(test))
	
	print("Mean accuracy: {:.2f}%".format(100 * np.mean(accuracies)))
	
	ids, gold_labels, pred_labels, pos_pred_probs = predict(test)
	pandas.DataFrame({"ID": ids,
				  "Class": gold_labels,
				  "Predicted Class": pred_labels,
				  "Predicted Probability": pos_pred_probs}).to_csv("predictions_NB_TextBlob.csv")
	
	#pandas.DataFrame({"Accuracy": accuracies}).to_csv("textblob_NB_accuracies.csv")