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


base_path = "log_19_07_22_0/"
model_path = base_path + "saved_model/0/tf_model/"



def predict_seq():

    # read test sequences
    path_test_data = base_path + "saved_data/test.h5"
    file_obj = h5py.File(path_test_data, 'r')
    test_input = tf.convert_to_tensor(np.array(file_obj["input"]), dtype=tf.int64)
    test_target = tf.convert_to_tensor(np.array(file_obj["target"]), dtype=tf.int64)
    print("Test data...")
    print(test_input)
    print(test_target)

    r_dict = utils.read_file(base_path + "data/rev_dict.txt")
    f_dict = utils.read_file(base_path + "data/f_dict.txt")
    
    tf_loaded_model = tf.saved_model.load(model_path)
    #predictor = predict_sequences.PredictSequence(tf_loaded_model)

    #predictor(test_input, test_target, f_dict, r_dict)

    tool_name = "cutadapt"
    print("Prediction for {}...".format(tool_name))
    bowtie_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    bowtie_output = bowtie_output.write(0, [tf.constant(index_start_token, dtype=tf.int64)])
    #bowtie_output = bowtie_output.write(1, [tf.constant(295, dtype=tf.int64)])
    bowtie_o = tf.transpose(bowtie_output.stack())

    tool_id = f_dict[tool_name]
    print(tool_name, tool_id)
    bowtie_input = np.zeros([1, 25])
    bowtie_input[:, 0] = index_start_token
    bowtie_input[:, 1] = tool_id
    bowtie_input[:, 2] = 1939
    #bowtie_input[:, 3] = 2141
    #bowtie_input[:, 4] = 1569 
    bowtie_input = tf.constant(bowtie_input, dtype=tf.int64)
    print(bowtie_input, bowtie_output, bowtie_o)
    bowtie_pred, _ = tf_loaded_model([bowtie_input, bowtie_o], training=False)
    print(bowtie_pred.shape)
    top_k = tf.math.top_k(bowtie_pred, k=10)
    print("Top k: ", bowtie_pred.shape, top_k, top_k.indices)
    print(np.all(top_k.indices.numpy(), axis=-1))
    print("Predicted next tools for {}: {}".format(tool_name, [r_dict[str(item)] for item in top_k.indices.numpy()[0][0]]))
    print()



if __name__ == "__main__":
    predict_seq()
