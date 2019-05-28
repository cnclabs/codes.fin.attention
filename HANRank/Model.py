import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
import random
import collections
import scipy.stats as  stats
import sys,os
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import Dense, Input, Embedding, Activation, MaxPooling1D, Embedding, GRU, Bidirectional, TimeDistributed, Subtract
from keras.models import Model, load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import optimizers, initializers
MAX_SENT_LENGTH = 70
MAX_SENTS = 150
EMBEDDING_DIM = 300
YEAR = int(sys.argv[1])

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
        weighted_input = x*ait
        output = K.sum(weighted_input, axis=1)

        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

mon = eval(open("./data/mon.txt").read())
data =  np.loadtxt('./data/'+str(YEAR)+'_data.txt')
data = data.reshape(-1,MAX_SENTS,MAX_SENT_LENGTH)
data = data.astype(int)
print(mon)
print(data.shape)


labels = np.loadtxt('./data/'+str(YEAR)+'_test.txt')
labels = labels.astype(int)
labels = labels.tolist()
print(len(labels))

word_index = eval(open("./data/"+str(YEAR)+"wordindex.txt").read())
print(len(word_index))
nb_validation_samples = mon[str(YEAR)]-mon[str(YEAR-1)]

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

GLOVE_DIR = "."
embeddings_index = {}
f = open(os.path.join('./word2vec/'+str(YEAR)+'word2vec.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# building Hierachical Attention network
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer()(l_lstm)
sentEncoder = Model(sentence_input, l_att)

sentEncoder.summary()

review_input = Input(shape=(None, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer()(l_lstm_sent)
preds = Dense(1)(l_att_sent)
model = Model(review_input, preds)
model.summary()

#building PairWise Ranking Model (hanrank)
review_input1 = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_input2 = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
output1 = model(review_input1)
output2 = model(review_input2)

output = Subtract()([output1,output2])
prob = Activation("sigmoid")(output)

pairWiseModel = Model(inputs=[review_input1,review_input2],output=prob)
pairWiseModel.summary()


optimizer =optimizers.Adam(lr=0.001,decay =10e-8)
pairWiseModel.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['acc'])

#Sampling & Training

for epoch in range(30):
    new_a=[]
    new_b=[]
    new_label=[]
    year= random.randint(YEAR-5, YEAR-1)
    if year==1996:
        head = 0
    else:
        head = mon[str(year-1)]-mon[str(YEAR-6)]
    tail = mon[str(year)]-mon[str(YEAR-6)]
    print(year,head,tail)
    while len(new_a)<3000:
        a,b= random.sample(range(head, tail),2)
        if y_train[a] is not y_train[b]:
            if (y_train[a]-y_train[b])>=1:
                 new_a.append(x_train[a])
                 new_b.append(x_train[b])
                 new_label.append(1)
            elif(y_train[b]-y_train[a])>=1:
                 new_a.append(x_train[a])
                 new_b.append(x_train[b])
                 new_label.append(0)
    print(len(new_a))
    new_a = np.array(new_a)
    new_b = np.array(new_b)
    
    print("model fitting - Hierachical attention network")
    
    pairWiseModel.fit([new_a,new_b],new_label,initial_epoch=epoch,batch_size=10, epochs=epoch+1)


#Vaildate pairWiseModel (Kendall's Tau, Spearman's Rho)
    get_score = K.function([pairWiseModel.get_layer('model_2').get_layer('input_2').input],[pairWiseModel.get_layer('model_2').get_layer('dense_1').output])
    
    score_list=[]
    for v in range (0,len(x_val)):
        score = get_score([[x_val[v]]])
        score_list.append(score[0][0][0])
    
    pred = stats.rankdata(score_list)
    true = y_val
    tau, p_value = stats.kendalltau(true, pred)
    spe, p_values = stats.spearmanr(true, pred)
    print(tau,spe)
    
#Save pairWiseModel
pairWiseModel.save('hanrank'+str(YEAR)+'.h5')
