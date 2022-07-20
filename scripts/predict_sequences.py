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



class PredictSequence(tf.Module):
  def __init__(self, trained_model):
    #self.tokenizers = tokenizers
    self.trained_model = trained_model


  def __call__(self, te_inp, te_tar, f_dict, r_dict):
    # input sentence is portuguese, hence adding the start and end token
    te_f_input = tf.constant(te_inp[0])
    te_f_input = tf.reshape(te_f_input, [1, max_seq_len])
    sentence = te_f_input
    start_token = index_start_token
    #print(te_f_input)
    #print(sentence)
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    encoder_input = sentence
    print("True input seq: ", encoder_input)

    print()

    # loop over target
    target_seq = te_tar[0]
    n_target_items = np.where(target_seq > 0)[0]
    np_output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    np_output_array = np_output_array.write(0, [tf.constant(start_token, dtype=tf.int64)])
    #print(np_output_array)
    te_tar_real = target_seq[1:]

    #print(n_target_items)
    print("Looping predictions ...")
    print("true target seq real: ", te_tar_real)
    for i, index in enumerate(n_target_items):
        #print(i, index)
        output = tf.transpose(np_output_array.stack())
        #print("decoder input: ", output, encoder_input.shape, output.shape)
        orig_predictions, _ = self.trained_model([encoder_input, output], training=False)
        #print(orig_predictions.shape)
        #print("true target seq real: ", te_tar_real)
        #print("Pred seq argmax: ", tf.argmax(orig_predictions, axis=-1))
        predictions = orig_predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        np_output_array = np_output_array.write(i+1, predicted_id[0])
        #print("----------")
    print(np_output_array)
    print("----------")
    '''_, attention_weights = self.trained_model([encoder_input, output[:,:-1]], training=False)
    print("Attention weights")
    print(attention_weights.shape)'''

    # predict for tool
    tool_name = "trim_galore"
    print("Prediction for {}...".format(tool_name))
    bowtie_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    bowtie_output = bowtie_output.write(0, [tf.constant(start_token, dtype=tf.int64)])
    bowtie_o = tf.transpose(bowtie_output.stack())

    tool_id = f_dict[tool_name]
    print(tool_name, tool_id)
    bowtie_input = np.zeros([1, 25])
    bowtie_input[:, 0] = index_start_token
    bowtie_input[:, 1] = tool_id
    bowtie_input[:, 2] = f_dict["bowtie2"]
    bowtie_input = tf.constant(bowtie_input, dtype=tf.int64)
    print(bowtie_input, bowtie_output, bowtie_o)
    bowtie_pred, _ = self.trained_model([bowtie_input, bowtie_o], training=False)
    print(bowtie_pred.shape)
    top_k = tf.math.top_k(bowtie_pred, k=10)
    print("Top k: ", bowtie_pred.shape, top_k, top_k.indices)
    print(np.all(top_k.indices.numpy(), axis=-1))
    print("Predicted next tools for {}: {}".format(tool_name, [r_dict[item] for item in top_k.indices.numpy()[0][0]]))
    print()
