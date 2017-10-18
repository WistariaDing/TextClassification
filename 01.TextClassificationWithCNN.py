# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:12:40 2017

Data is from the reviews of movies in 'data/labeledTrainData.tsv'
Two Model: 
    1. Common CNN Model
    2. Complex CNN Model from Yoon Kim's Paper. Merge multiple filters.
    
It proves the second model preforms better.

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

import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense,Input,Flatten
from keras.layers import Conv1D, MaxPooling1D,Dropout,Concatenate
from keras.models import Model, Sequential

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


model = Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(35))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - convolutional 1D neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=128)


#-----------------------Complex CNN ------------------------------------
"""
In Yoon Kim’s paper, multiple filters have been applied.
"""
print ('---Start to run Complex CNN model--------------:')

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

convs = []
filter_sizes = [3,4,5]

for fsz in filter_sizes:
    l_conv = Conv1D(filters=128,kernel_size=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)

l_merge = Concatenate(axis=1)(convs)
l_cov1 = Conv1D(128,5,activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128,5,activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128,activation='relu')(l_flat)
out = Dense(2,activation='softmax')(l_dense)

model2 = Model(sequence_input,out)
model2.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


model2.summary()
print("model fitting - complex CNN network")
model2.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=50)










