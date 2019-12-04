'''
	TEST code from Medium article to create data via SMOTE algorithm
'''

import numpy as np
import pandas
from random import sample, random
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
import re
from math import sqrt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# import tensorflow
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical

from scipy.sparse import hstack

#from seaborn import heatmap

def build_classifier(X_train, y_train, X_test, y_test):
	# Create the model
	model = Sequential()
	model.add(Dense(20, input_dim=X_train.shape[1], activation="relu"))
	model.add(Dense(20, activation="relu"))
	model.add(Dense(3, activation="softmax")) 

	# Compile the model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Train the model
	history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

	print(history)
	print(history.history)
	print(history.history['accuracy'])

	# Plot accuracy
	def plot_history(history, key='accuracy'):
		plt.figure(figsize=(16,10))

		val = plt.plot(history.epoch, history.history['val_'+key], '--', label='Accuracy on validation set')
		plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=' Accuracy on training set')

		plt.xlabel('Epochs')
		plt.ylabel(key.replace('_',' ').title())
		plt.legend()

		plt.xlim([0,max(history.epoch)])

		plt.show() 
		
	plot_history(history)
	return model


def test_classifier(model, X_test, y_test):
	# Evaluate the model
	y_pred = model.predict(X_test)
	metrics = model.evaluate(X_test, y_test)

	# Determine category from One Hot Encoding
	categorize = lambda encoded: [ argmax(line) for line in encoded ]
	y_pred = categorize(y_pred)
	y_test = categorize(y_test)

	print('=============================')
	print(y_pred)
	print(y_test)
	print('=============================')

	# Plot confusion matrix
	cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
	print(cf_matrix)
	heatmap(cf_matrix, cmap='Blues')
	plt.show()

	# Print metrics
	for i, metric in enumerate(metrics):
		print(model.metrics_names[i], '=', metric)



def open_doc(path,sepa):
	return pandas.read_csv(path, sep=sepa)

def nearest_neighbour(X, x):
    # Compute euclidian distance to find nearest neighbours
    euclidean = np.ones(X.shape[0]-1)
    
    additive = [None]*(1*X.shape[1])
    additive = np.array(additive).reshape(1, X.shape[1])
    k = 0
    for j in range(0,X.shape[0]):
        if np.array_equal(X[j], x) == False:
            euclidean[k] = sqrt(sum((X[j]-x)**2))
            k = k + 1
    euclidean = np.sort(euclidean)
    weight = random()
    while(weight == 0):
        weight = random()
    additive = np.multiply(euclidean[:1],weight)
    return additive
    
def SMOTE_100(X):
    new = [None]*(X.shape[0]*X.shape[1])
    new = np.array(new).reshape(X.shape[0],X.shape[1])
    k = 0
    for i in range(0,X.shape[0]):
        additive = nearest_neighbour(X, X[i])
        for j in range(0,1):
            new[k] = X[i] + additive[j]
            k = k + 1
    return new # the synthetic samples created by SMOTe 

# Get data
reader = open_doc("./osna/data_collection/commentssarc.csv",';')
sentiments = open_doc("./osna/sentiment_analysis/sentiments_usingtextblob.csv",',')
reader['polarity'] = sentiments['polarity']
reader['subjectivity'] = sentiments['subjectivity']
features = ['body', 'polarity', 'subjectivity', 'sarcarsm', 'label']
data = reader[features]

# Select as much data for each label
minus_ones = np.array([ line for line in data.values if line[-1] == -1 ])
ones = np.array(sample([ line for line in data.values if line[-1] == 1 ], len(minus_ones)))
zeros = np.array(sample([ line for line in data.values if line[-1] == 0 ], len(minus_ones)))
data = np.vstack([minus_ones, ones, zeros])

# Make feature matrix
y_data = data[:,-1]
X_data = data[:,:-1]
tfidf_comment = TfidfVectorizer(ngram_range=(1,2), max_features=None)
comment_sparse = tfidf_comment.fit_transform(X_data[:,0])
comment_array = comment_sparse.toarray()

# Divide data into train and test
X_train, X_test, y_train, y_test = train_test_split(comment_array, y_data, test_size=0.2, random_state=1234)
X_train, X_v, y_train, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=2341)

unique, counts = np.unique(y_train, return_counts=True)
minority_shape = dict(zip(unique, counts))[1]

# Storing the minority class instances separately
x1 = np.ones((minority_shape, X_train.shape[1]))
k=0
for i in range(0,X_train.shape[0]):
    if y_train[i] == 1.0:
        x1[k] = X_train[i]
        k = k + 1
# Applying 100% SMOTe
sampled_instances = SMOTE_100(x1)

# Keeping the artificial instances and original instances together
X_f = np.concatenate((X_train,sampled_instances), axis = 0)
y_sampled_instances = np.ones(minority_shape)
y_f = np.concatenate((y_train,y_sampled_instances), axis=0)
# X_f and y_f are the Training Set Features and Labels respectively

print(len(X_train), len(sampled_instances), len(X_f))
print(X_train)
print('=============================')
print(X_f)

# model = build_classifier(X_f, y_f, X_test, y_test)
# test_classifier(model, X_test, y_test)

