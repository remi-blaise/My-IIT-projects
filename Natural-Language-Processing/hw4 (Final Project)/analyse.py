#!/usr/bin/python3

import pickle
from collections import Counter
from itertools import product
from prettytable import PrettyTable

from classifiers.read_data import read_data
from classifiers.models import *

# Retrieve classifier
with open('best_classifier', 'rb') as file:
	model = pickle.load(file)

# Read data
X_eval, y_eval = read_data(dataset='eval')

# Get predictions
prediction = model.predict(X_eval)

# Compute stats

classes = sorted(set(y_eval))
stats = {
	'class': classes,
	'total_expected': [0] * len(classes),
	'total_predicted': [0] * len(classes),
	'true_positive': [0] * len(classes),
	'false_positive': [0] * len(classes),
	'false_negative': [0] * len(classes),
	'most_misclassified_as': [None] * len(classes),
	'precision_measure': [None] * len(classes),
	'recall_mesure': [None] * len(classes),
	'f1_measure': [None] * len(classes),
}
counters = [ Counter() for x in classes ]

for expected, predicted in zip(y_eval, prediction):
	i = classes.index(expected)
	j = classes.index(predicted)

	stats['total_expected'][i] += 1
	stats['total_predicted'][j] += 1

	if expected == predicted:
		stats['true_positive'][i] += 1
	else:
		stats['false_negative'][i] += 1
		stats['false_positive'][j] += 1

	counters[i].update([predicted])

for i, expected in enumerate(classes):
	stats['most_misclassified_as'][i] = [ x for x, n in counters[i].most_common() if x != expected ][0]
	stats['precision_measure'][i] = stats['true_positive'][i] / ( stats['true_positive'][i] + stats['false_positive'][i] )
	stats['recall_mesure'][i] = stats['true_positive'][i] / ( stats['true_positive'][i] + stats['false_negative'][i] )
	stats['f1_measure'][i] = 2 * stats['precision_measure'][i] * stats['recall_mesure'][i] / ( stats['precision_measure'][i] + stats['recall_mesure'][i] )

misclassifications = [] # { (x, y): 0 for x, y in product(classes, classes) }

for i, expected in enumerate(classes):
	for predicted in classes:
		misclassifications.append(((expected, predicted), counters[i][predicted]))

# Print stats

stats['precision_measure'] = list(map(lambda x: '%.3f' % x, stats['precision_measure']))
stats['recall_mesure'] = list(map(lambda x: '%.3f' % x, stats['recall_mesure']))
stats['f1_measure'] = list(map(lambda x: '%.3f' % x, stats['f1_measure']))

print("Stats by class:")

headers = list(stats.keys())
columns = list(stats.values())
rows = [ [ column[i] for column in columns ] for i in range(len(classes)) ]
table = PrettyTable(headers)
for row in rows:
	table.add_row(row)
print(table)
print()

print("Classifications by pair:")

table = PrettyTable(["classes", "number of classified"])
for c, m in misclassifications:
	table.add_row(["%s as %s" % c, m])
print(table)
print()
