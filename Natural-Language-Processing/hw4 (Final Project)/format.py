#!/usr/bin/python3

import csv
import os

DATA_DIR = 'data/'
OUTPUT_DIR = 'bert_input_data/'

for filename in os.listdir(DATA_DIR):
	with open(DATA_DIR + filename) as csvFile:
		with open(OUTPUT_DIR + filename[8:-4] + '.txt', mode='w') as outputFile:
			reader = csv.reader(csvFile)
			reader.__next__()
			for nativeLanguage, text in reader:
				print(text, file=outputFile)
