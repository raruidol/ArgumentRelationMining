import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM, Bidirectional, SimpleRNN, GRU
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import SGD, Adam, RMSprop
from scipy import stats
from collections import Counter
from copy import deepcopy

def recurrent_neural_network(out_size):
    rnn = Sequential()

    rnn.add(SimpleRNN(128, input_shape=(1, 5)))

    rnn.add(Dense(out_size, activation='sigmoid'))

    #rnn.summary()
    rnn.compile('adam', loss=corr_loss, metrics=[corr])

    return rnn


def long_short_term_memory(out_size):
    lstm = Sequential()

    lstm.add(LSTM(512, input_shape=(1, 5)))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(out_size, activation='sigmoid'))

    #lstm.summary()
    lstm.compile('adam', loss=corr_loss, metrics=[corr])

    return lstm


history = model.fit(x_tr, y_tr,
                                batch_size=319,
                                epochs=20,
                                validation_split=0.05,
                                verbose=0)

