import collections
import json
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
import random
from bs4 import BeautifulSoup
import collections
from nltk import tokenize
from nltk.corpus import stopwords
import scipy.stats as  stats
import sys
import os
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
MAX_SENT_LENGTH = 70
MAX_SENTS = 150
EMBEDDING_DIM = 300
YEAR= int(sys.argv[1])


def clean_str(string):
    string = re.sub(r"\\", "", str(string))
    string = re.sub(r"\'", "", str(string))
    string = re.sub(r"\"", "", str(string))
    return string.strip().lower()

if not os.path.exists('./data'):
    os.makedirs('./data')

data_train = pd.read_pickle('./data.pkl')

#Make data index of each year from data.pkl
d =  data_train['year'].value_counts(sort=False).to_dict()
sum = 0
sumlist=[]
for k,v in d.items():
    sum = sum+v
    d[k]=sum
d[1995]=0
with open('./data/mon.txt', 'w') as file:
     file.write(json.dumps(d))


data_train = data_train[data_train['year']>=(YEAR-5)]
data_train = data_train[data_train['year']<(YEAR+1)]
print (YEAR, data_train.shape)

reviews = []
labels = []
texts = []
for idx in range(data_train.porter_stop.shape[0]):
    text = BeautifulSoup(data_train.porter_stop.iloc[idx])
    text = clean_str(data_train.porter_stop.iloc[idx].encode('ascii', 'ignore'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)
    labels.append(data_train.labelmark.iloc[idx])
print(len(reviews),len(labels),len(texts),idx)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index
with open('./data/'+str(YEAR)+'wordindex.txt', 'w') as file:
    file.write(json.dumps(word_index))

print('Total %s unique tokens.' % len(word_index))
print('Shape of data tensor:', data.shape)
    
data = data.reshape(-1,MAX_SENT_LENGTH)
np.savetxt("./data/"+str(YEAR)+"_data.txt",data)
np.savetxt("./data/"+str(YEAR)+"_test.txt",labels)


