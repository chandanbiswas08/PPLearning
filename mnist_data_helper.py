import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tf.keras.backend.set_floatx('float64')
from sklearn.externals import joblib
import sys
import pickle
from numpy.random import seed
import math
from shutil import copyfile
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import gc

# from Config import *
import Config
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
seed(1234)
tf.random.set_seed(2)


def writeVectors(vec=None, vec_file='data/vecs.txt'):
    print("Writing vectors...")
    np.savetxt(vec_file, np.asarray(vec))


def loadInputVec(path):
    input_vec=[]
    with open(path) as fp:
        for line in fp:
            line1=line.split('\n')
            words = line1[0].strip(" ").split(' ')
            z=[]
            for word in words:
                z.append(float(word))
            input_vec.append(z)
    input_vec = np.asarray(input_vec, dtype="float32")
    return input_vec


def load_labels(path):
    labels=[]
    with open(path) as fp:
        for line in fp:
            line1=line.split('\n')
            labels.append(int(line1[0]))
    labels = np.asarray(labels)
    num_class = int(max(labels)) + 1
    return labels, num_class


def make_onehot(labels, num_class):
    output=[]
    for i in range(len(labels)):
        x=np.zeros(num_class)
        x[labels[i]] = 1
        output.append(x)
    return output


def get_batches(data, batch_size=32, tao=0.2):
    data_x, data_p = data[0], data[1]
    total = len(data_x)
    row_count = np.shape(data_x)[1]
    imp_row_count = math.ceil(tao * row_count)
    start = 0
    while(1):
        end = min(start+batch_size,total)
        batch_x = data_x[start:end]
        batch_p = data_p[start:end]
        batch_mask = np.zeros((end - start,row_count))
        batch_mask[np.sum(batch_x, axis=-1) > 0 ] = 1
        yield batch_x, batch_p, batch_mask, start, end, imp_row_count, row_count
        start = end
        if(start >= total):
            start = 0

