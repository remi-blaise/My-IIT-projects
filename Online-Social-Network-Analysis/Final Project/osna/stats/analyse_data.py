from collections import Counter, OrderedDict
import pandas
import numpy as np
import matplotlib.pylab as plt
import re

DATA_PATH = './osna/data_collection/'
STAT_PATH = './osna/stats/'

def read_train_comments():
    '''Get comments'''
    table = pandas.read_csv(DATA_PATH + 'comments2.csv', sep=';')
    print(table)
    table2 = pandas.read_csv(DATA_PATH + 'comments.csv', sep=';')
    print(table2)
    table3 = pandas.read_csv(DATA_PATH + 'comments3.csv', sep=',')
    print(table3)
    table=table[:1000]
    table2=table2[:1000]
    table3=table3[1001:2000]
    tables=[table, table3,table2]
    table=pandas.concat(tables)
    
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

def analyse_data(table, showMode=False, img_prefix=''):
    '''Compute number of comments for each class, total number of articles, number of authors and number of tokens in data
	   Plot claim depending on non-believers and authors depending on number of comments
	'''
    labels = Counter()
    articles = Counter()
    authors = Counter()
    count_tokens = Counter()

    for (i,l) in enumerate(table['label']):
        if not pandas.isna(l):
            labels[l] +=1
            if l == -1:
                articles[table.iloc[i]['claim_id']] += 1
            if not pandas.isna(table.iloc[i]['author']):
                authors[table.iloc[i]['author']] += 1

    for i, comment in enumerate(table['body']):
	    count_tokens.update(tokenize(comment))

    if showMode:
        print('For category "non-believer", we have', labels[-1], 'comments.')
        print('For category "believer", we have', labels[1], 'comments.')
        print('For category "none of the above", we have', labels[0], 'comments.')

        print('Number of labelized articles', len(articles))
        print('Number of authors on labelized data', len(authors))
        print('Number of tokens in data', len(count_tokens))

    nonBelievers = OrderedDict(sorted(articles.items(), key=lambda pair: pair[1], reverse=True))
    authors = OrderedDict(sorted(authors.items(), key=lambda pair: pair[1], reverse=True))

    plt.figure()
    plt.xlabel('Claims', fontsize=15)
    plt.ylabel('Number of non-believers', fontsize=15)
    plt.plot(range(len(nonBelievers)), list(nonBelievers.values()))
    plt.xticks([])
    plt.savefig(STAT_PATH + img_prefix + 'nonBelievers.png')
    if showMode:
        plt.show()

    plt.figure()
    plt.xlabel('Authors', fontsize=15)
    plt.ylabel('Number of comments', fontsize=15)
    plt.bar(list(authors.keys()), list(authors.values()), width=.9)
    plt.title("Author histogram")
    plt.xticks([])
    plt.savefig(STAT_PATH + img_prefix + 'comments.png')
    if showMode:
        plt.show()

    return labels, articles, authors, count_tokens
