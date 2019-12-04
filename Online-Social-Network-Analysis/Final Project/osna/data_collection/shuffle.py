# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
PATH = 'osna/data_collection/'

# Get comments
reader = pd.read_csv(PATH + 'comments_collected.csv', delimiter=',')
reader.info()
# Removed 1st 1000 comments because they were already labellized
reader=reader[1000:]
reader.info()
# Shuffle all comments and put the result in a csv file
reader=shuffle(reader)
pd.DataFrame(reader).to_csv(PATH + 'comments_collected.csv', sep=',')
# pd.DataFrame(reader).to_csv(PATH + 'comments_sarcasm.csv', sep=';')
