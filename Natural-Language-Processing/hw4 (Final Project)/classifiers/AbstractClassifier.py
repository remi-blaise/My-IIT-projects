#!/usr/bin/python3

from abc import ABC, abstractmethod



class AbstractClassifier(ABC):
	'''An abstract class that defines classifiers' protocol'''


	@abstractmethod
	def build(self, X, y, params):
		pass


	@abstractmethod
	def predict(self, X):
		pass


	@abstractmethod
	def evaluate(self, X, y):
		pass


	@staticmethod
	@abstractmethod
	def exploration_set():
		pass


	@staticmethod
	@abstractmethod
	def exploration_log():
		pass


	@classmethod
	def log(cls, params, score):
		s = cls.exploration_log() % params
		print(s + " "*(105-len(s)) + " => %f" % score, flush=True)


	def explore(self, X_train, y_train, X_test, y_test):
		'''Explore models on exploration_set() and return the best'''

		best = None
		best_accuracy = 0

		for params in self.exploration_set():
			self.build(X_train, y_train, params)
			score = self.evaluate(X_test, y_test)
			self.log(params, score)

			if best_accuracy < score:
				best = params
				best_accuracy = score

		return best, best_accuracy


