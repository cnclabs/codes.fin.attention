import pandas as pd
from nltk.corpus import stopwords
from multiprocessing import Pool
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os, sys
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Activation, MaxPooling1D, Embedding, GRU, Bidirectional, TimeDistributed, Subtract
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers, initializers
from keras.engine.topology import Layer, InputSpec

MAX_SENT_LENGTH = 70
MAX_SENTS = 150
EMBEDDING_DIM = 300

class AttLayer(Layer):
    def __init__(self,**kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = 100
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        self.aitt = ait
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def preprocessing(text):
    
	text = re.sub('        ','', text)
	text =  re.sub('-','', text)
	text =  re.sub('\n',' ', text)
	text =  re.sub('<PAGE>',' ', text)
	text =  re.sub('_',' ',text)
	text =  re.sub('\s{2,}',' ', text)
	text =  re.sub('Mr. ','Mr.', text)
	text =  re.sub('Ms. ','Ms.', text)
	text =  re.sub('Mrs. ','Mrs.', text)
	sent_list  = text.split('. ')
	fin_text = ''
	stopWords = set(stopwords.words('english'))
	porter_stemmer = PorterStemmer()
	for sent in sent_list:
		s = re.sub('\W', ' ',sent)
		s = re.sub('_',' ',s)
		s = re.sub('Table of Contents','',s)
		s = re.sub('\d','',s)
		s = re.sub('\s{2,}',' ', s)
		s = s.strip()
		s = s.lower()
		#filter sents containing less 4 words 
		if len(s)>4:
			words = s.split(' ')
			for word in words:
				#print(word)
				if word not in stopWords or word=='.':
					#filter char
					if len(word)>1:
						word = porter_stemmer.stem(word)
						fin_text = fin_text + word + ' '
			fin_text = fin_text.strip()		
			fin_text += '. '
	fin_text = fin_text.strip()
	return fin_text


def create_word_index(text):
	word_index = eval(open("./2001wordindex.txt").read())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(text[0])
	data = np.zeros((len(text), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

	for i, sentences in enumerate(text):
		for j, sent in enumerate(sentences):
			if j < MAX_SENTS:
				wordTokens = text_to_word_sequence(sent)
				k = 0
				for _, word in enumerate(wordTokens):
					if k < MAX_SENT_LENGTH:
						data[i, j, k] = word_index[word]
						k = k + 1

	return word_index, data

def get_word_attention(model, data):

    res = dict((v,k) for k,v in word_index.items())
    get_weight = K.function([model.get_layer('model_2').get_layer('time_distributed_1').layer.get_layer('input_1').input],[model.get_layer('model_2').get_layer('time_distributed_1').layer.get_layer('bidirectional_1').output])	
    
    WW = model.get_layer('model_2').get_layer('time_distributed_1').layer.layers[3].get_weights()[0]
    bb = model.get_layer('model_2').get_layer('time_distributed_1').layer.layers[3].get_weights()[1]
    uu = model.get_layer('model_2').get_layer('time_distributed_1').layer.layers[3].get_weights()[2]
    
    for index, element in enumerate(data):
        att=np.dot(np.tanh(np.dot(get_weight([data[index]])[0],WW)+bb),uu)
        soft=np.exp(np.squeeze(att).T)
        attention = soft/soft.sum(0)
        attention = attention.T
        element = np.vectorize(res.get)(element)
    
    print(attention)
    print(element)
    
    return attention, element

def get_sent_attention(model, data):

    get_att_fun=K.function([model.get_layer('input_4').input],[model.get_layer('model_2').get_layer('att_layer_2').aitt])
    attention = np.squeeze(get_att_fun([data])[0])
    print(attention)
    return attention, element

#Read file
f = open('./989922109-10-K-19961227.mda',"r") # Read a financial report
text  =''
while True:
	line = f.readline()
	text = text + line
	if not line:
		break
#preprocessing
print('start preprocessing')
fin_text = preprocessing(text)
#create_word_index
sentences = tokenize.sent_tokenize(fin_text)
fin_text = [sentences]
word_index, data = create_word_index(fin_text)
#load model
print('start loading model')
model = load_model('./hanrank2001.h5',custom_objects={'AttLayer':AttLayer}) # load model
#get attention weight
print('start calculating attention')
word_attention, element = get_word_attention(model, data) #get each word attenation value
sent_attention, element = get_sent_attention(model, data) #get each sentence attention value

