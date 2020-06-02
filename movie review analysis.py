# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:53:29 2020

@author: SHASHANK RAJPUT
"""
# importing some packages
import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation
from keras.preprocessing.text import Tokenizer


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import re
import string

# For reproducibility

from numpy.random import seed
seed(1)


LABELS = ['negative', 'positive']
train = pd.read_csv('train.tsv', sep='\t') 
val = pd.read_csv("test.tsv", sep='\t') 

print(train.shape)
print(val.shape)
print(train['Phrase'][0])


# Custom Tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    
    return re_tok.sub(r' \1 ', s).split()

imdb_tokenizer = Tokenizer(num_words=30000)
imdb_tokenizer.fit_on_texts(train['Phrase'].values)

x_train_seq = imdb_tokenizer.texts_to_sequences(train['Phrase'].values)
x_val_seq = imdb_tokenizer.texts_to_sequences(val['Phrase'].values)

MAX_LEN=500
x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", value=0)
x_val = sequence.pad_sequences(x_val_seq, maxlen=MAX_LEN, padding="post", value=0)

y_train, y_val = train['PhraseId'].values, val['PhraseId'].values

print('First sample before preprocessing: \n', train['Phrase'].values[28], '\n')
print('First sample after preprocessing: \n', x_train[28])

#CNN MODEL
model = Sequential()


model.add(Embedding(30000, 40, input_length=500))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn NUM_FILTERS filters
model.add(Conv1D(250,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))

# we use max pooling:
model.add(GlobalMaxPooling1D())

model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))


model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit a model
model.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          validation_split=0.1,
          verbose=2)

# Evaluate the model
score, acc = model.evaluate(x_val, y_val, batch_size=128)
print('\nAccuracy: ', acc*100)

pred = model.predict_classes(x_val)
#TRY WITH OWN RIVOEW

from ipywidgets import interact_manual
from ipywidgets import widgets

def get_prediction(review):
    # Preprocessing
    review_np_array = imdb_tokenizer.texts_to_sequences([review])
    review_np_array = sequence.pad_sequences(review_np_array, maxlen=500, padding="post", value=0)
    # Prediction
    score = model.predict(review_np_array)[0][0]
    prediction = LABELS[model.predict_classes(review_np_array)[0][0]]
    print('REVIEW:', review, '\nPREDICTION:', prediction, '\nSCORE: ', score)

interact_manual(get_prediction, review=widgets.Textarea(placeholder=' Quentin Tarantino proves that he is the master of witty dialogue and a fast plot that doesn\'t allow the viewer a moment of boredom or rest.'));


