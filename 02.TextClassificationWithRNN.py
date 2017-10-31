# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:39:06 2017
In this post, the model is based on recurrent neural network and attention based LSTM encoder.
The attention network is implemented on top of LSTM for the classification task.
@author: teding
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model, Sequential

from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras import backend as K

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\","",string)
    string = re.sub(r"\'","",string)
    string = re.sub(r"\"","",string)
    return string.strip().lower()

# Parameters setting
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# Data input
data_train = pd.read_csv('data/labeledTrainData.tsv',sep='\t')

texts=[]
labels=[]

# Use BeautifulSoup to remove some html tags and remove some unwanted characters.
for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx],'lxml')
    texts.append(clean_str(text.get_text()))
    labels.append(data_train.sentiment[idx])

tokenizer=Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
labels = np.asarray(labels, dtype = np.float32)
print('Shape of data tensor: ', data.shape)
print('Shape of label tensor: ',labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print ('Number of negative and positive reviews in training and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

## Use pre-trained wordToVec

embeddings_index = {}
f=open('data/glove.6B/glove.6B.100d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))


embedding_matrix = np.random.random((len(word_index)+1,EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector


#create model
model = Sequential()
model.add(Embedding(len(word_index) +1,
                    EMBEDDING_DIM,
                    weights = [embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable = True))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()

print("model fitting - Bidirectional LSTM")

model.fit(x_train, y_train, 
          validation_data=(x_val,y_val),
          epochs=1,batch_size=50)


#-------------------------Attention model---------------

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(** kwargs)
    
    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.init((input_shape[-1],1))
        self.trainable_weights=[self.W]
        super(AttLayer,self).build(input_shape)
    def call(self,x,mask=None):
        eij = K.tanh(K.dot(x,self.W))
#        print('W shape:',self.W.shape)
#        print('eij shape: ', eij.shape)
        ai = K.exp(eij)
#        print('aij shape: ',ai.shape)
        weights = ai/K.sum(ai,axis=1)
#        print('weigths shape: ',weights.shape)
#        print('x shape: ',x.shape)
        weighted_input = x*weights
#        print('input shape: ', weighted_input.shape)
        output=K.sum(weighted_input,axis=1)
        
        return output
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])



embedding_layer = Embedding(len(word_index) +1,
                    EMBEDDING_DIM,
                    weights = [embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(GRU(100,return_sequences=True))(embedded_sequences)
l_att = AttLayer()(l_gru)
preds = Dense(2, activation='softmax')(l_att)
model = Model(sequence_input,preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()

print("model fitting - Attentional LSTM")

model.fit(x_train, y_train, 
          validation_data=(x_val,y_val),
          epochs=1,batch_size=50)







