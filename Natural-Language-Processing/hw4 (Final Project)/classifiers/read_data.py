#!/usr/bin/python3

import os
import json
import numpy as np
import pandas as pd
import pickle


CACHE_DIR = 'cache/'


def read_data(dataset='train'):
	'''Read features and labels of a given dataset. 

	The retrieved data is cached in order to avoid reading the heavy output of BERT at each execution.
	'''

	cache_filename = CACHE_DIR + dataset

	# If the data is cached, retrieve it
	try:
		with open(cache_filename, 'rb') as file:
			return pickle.load(file)

	# else, read BERT output data
	except FileNotFoundError:
		ORIGINAL_DATA_DIR = "data"
		BERT_FEATURE_DIR = "bert_output_data"

		df = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "lang_id_" + dataset + ".csv"))

		bert_vectors = []
		with open(os.path.join(BERT_FEATURE_DIR, dataset + ".jsonlines"), "rt") as infile:
		    for line in infile:
		        bert_data = json.loads(line)
		        for t in bert_data["features"]:
		            # Only extract the [CLS] vector used for classification
		            if t["token"] == "[CLS]":
		                # We only use the representation at the final layer of the network
		                bert_vectors.append(t["layers"][0]["values"])
		                break

		X = np.array(bert_vectors)
		y = df["native_language"].values

		# Save in cache
		with open(cache_filename, 'wb') as file:
			pickle.dump((X, y), file)

		return X, y


if __name__ == '__main__':
	X, y = read_data()
	print(type(X), X.shape)
	print(type(y), y.shape)
