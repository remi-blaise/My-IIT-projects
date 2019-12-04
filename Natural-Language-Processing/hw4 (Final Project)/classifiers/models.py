#!/usr/bin/python3

from abc import ABC, abstractmethod
from itertools import product
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.neighbors import KNeighborsClassifier as KNeighbors
from sklearn.neural_network import MLPClassifier

from .read_data import read_data
from .AbstractClassifier import AbstractClassifier


import warnings
warnings.filterwarnings("ignore")


CORES = 7



class BaseSklearnClassifier(AbstractClassifier, ABC):
	'''A base class for sklearn classifiers'''


	@abstractmethod
	def instanciate(self, params):
		pass


	def build(self, X, y, params):
		self.instanciate(params)
		self.model.fit(X, y)


	def predict(self, X):
		return self.model.predict(X)


	def evaluate(self, X, y):
		return self.model.score(X, y)



class LogisticRegressionClassifier(BaseSklearnClassifier):
	'''Use sklearn LogisticRegression model'''


	@staticmethod
	def exploration_set():
		return (('newton-cg',), ('lbfgs',), ('liblinear',), ('sag',), ('saga',))


	@staticmethod
	def exploration_log():
		return "LogisticRegression(solver=%s, penalty='l2')"


	def instanciate(self, params):
		solver, = params
		self.model = LogisticRegression(solver=solver, penalty='l2')



class SVMClassifier(BaseSklearnClassifier):
	'''Use sklearn SVM model'''


	@staticmethod
	def exploration_set():
		return ((),)


	@staticmethod
	def exploration_log():
		return "SVC()"


	def instanciate(self, params):
		self.model = SVC()



class RandomForestClassifier(BaseSklearnClassifier):
	'''Use sklearn RandomForest model'''


	@staticmethod
	def exploration_set():
		return ((10,), (100,), (1000,), (5000,), (10000,))


	@staticmethod
	def exploration_log():
		return "RandomForest(n_estimators=%d, n_jobs=CORES)"


	def instanciate(self, params):
		n_estimators, = params
		self.model = RandomForest(n_estimators=n_estimators, n_jobs=CORES)



class KNeighborsClassifier(BaseSklearnClassifier):
	'''Use sklearn KNeighbors model'''


	@staticmethod
	def exploration_set():
		return ((5,), (10,), (50,), (200,))


	@staticmethod
	def exploration_log():
		return "KNeighbors(n_neighbors=%d, n_jobs=CORES)"


	def instanciate(self, params):
		n_neighbors, = params
		self.model = KNeighbors(n_neighbors=n_neighbors, n_jobs=CORES)



class MultiLayerPerceptronClassifier(BaseSklearnClassifier):
	'''Use sklearn MLPClassifier model'''


	@staticmethod
	def exploration_set():
		return ((100, 5, 'adam'),)
		return product([ x*50 for x in range(1, 5) ], range(3, 10), ('lbfgs', 'sgd', 'adam'))


	@staticmethod
	def exploration_log():
		return "MLPClassifier(hidden_layer_sizes=(%d,)*%d, solver='%s',  activation='relu', early_stopping=True)"


	def instanciate(self, params):
		n_neurons, n_layers, solver = params
		self.model = MLPClassifier(hidden_layer_sizes=(n_neurons,)*n_layers, solver=solver,  activation='relu', early_stopping=True)



if __name__ == '__main__':
	import argparse

	# Read classifier from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('classifier', type=str)
	classifier_name = parser.parse_args().classifier

	if classifier_name == 'all':
		classifiers = [ LogisticRegressionClassifier(), SVMClassifier(), RandomForestClassifier(), KNeighborsClassifier(), MultiLayerPerceptronClassifier() ]
	else:
		classifiers = [ globals()[classifier_name]() ]

	# Read data
	X_train, y_train = read_data()
	X_test, y_test = read_data(dataset='test')
	X_eval, y_eval = read_data(dataset='eval')

	# Explore models
	best = None
	best_accuracy = 0

	for classifier in classifiers:
		print("\n" + "#"*120 + "\n", flush=True)

		params, score = classifier.explore(X_train, y_train, X_test, y_test)
		classifier.build(X_train, y_train, params)

		print("\nChoice: ", flush=True)
		classifier.log(params, score)
		print(flush=True)

		if best_accuracy < score:
			best = classifier
			best_accuracy = score

	# Evaluate model accuracy

	# Print result
	# prediction = best.predict(X_eval)
	# print("\nResult on eval set", flush=True)
	# for i in range(len(y_eval)):
	# 	print(prediction[i], y_eval[i], flush=True)
	# print(flush=True)

	print("\nBest: ", flush=True)
	best.log(params, score)
	print(flush=True)

	print("Accuracy on eval set: ", best.evaluate(X_eval, y_eval), flush=True)

	# Save the model
	with open('best_classifier', 'wb') as file:
		pickle.dump(best, file)
