"""
Create transformers
"""
import os
import time
import numpy as np
import subprocess
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding, SpatialDropout1D
#from tensorflow.keras.layers.embeddings import 
#from keras.layers.core import 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping


import onnx
from onnx_tf.backend import prepare

import predict_sequences
import utils


BATCH_SIZE = 64
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 20
max_seq_len = 25
index_start_token = 2


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

model_path = "log/saved_model/10/tf_model/"

def predict_seq():

    # read test sequences
    path_test_data = "log/saved_data/test.h5"
    file_obj = h5py.File(path_test_data, 'r')
    test_input = tf.convert_to_tensor(np.array(file_obj["input"]), dtype=tf.int64)
    test_target = tf.convert_to_tensor(np.array(file_obj["target"]), dtype=tf.int64)
    
    print(test_input)
    print(test_target)

    r_dict = utils.read_file("log/data/rev_dict.txt")
    f_dict = utils.read_file("log/data/f_dict.txt")
    
    tf_loaded_model = tf.saved_model.load(model_path)
    predictor = predict_sequences.PredictSequence(tf_loaded_model)

    predictor(test_input, test_target, f_dict, r_dict)



if __name__ == "__main__":
    predict_seq()
