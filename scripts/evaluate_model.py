"""
Create transformers
"""
import os
import time
import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import sys
import random

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


'''BATCH_SIZE = 64
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 20
max_seq_len = 25
index_start_token = 2'''

embed_dim = 128 # Embedding size for each token
num_heads = 8 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer
d_dim = 128
dropout = 0.1
n_train_batches = 1000000
batch_size = 32
test_logging_step = 100
train_logging_step = 2000
n_test_seqs = batch_size
learning_rate = 1e-2

fig_size = (12, 12)
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

'''loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')'''


base_path = "log_27_07_22_0/"
model_path = base_path + "saved_model/382000/tf_model/"


def plot_loss_acc(loss, acc, t_value):
    # plot training loss
    x_val = np.arange(len(loss))
    #x_val = np.arange(n_epo)
    plt.plot(x_val, loss)
    plt.ylabel("Loss".format(t_value))
    plt.xlabel("Training steps")
    plt.grid(True)
    plt.title("{} vs loss".format(t_value))
    plt.savefig("log/data/{}_loss.pdf".format(t_value), dpi=200)
    plt.show()

    # plot driver gene precision vs epochs
    x_val_acc = np.arange(len(acc))
    #x_val = np.arange(n_epo)
    plt.plot(x_val_acc, acc)
    plt.ylabel("Accuracy".format(t_value))
    plt.xlabel("Training steps")
    plt.grid(True)
    plt.title("{} steps vs accuracy".format(t_value))
    plt.savefig("log/data/{}_acc.pdf".format(t_value), dpi=200)
    plt.show()


def visualize_loss_acc():
    epo_tr_batch_loss = utils.read_file("log/data/epo_tr_batch_loss.txt").split(",")
    epo_tr_batch_loss = [np.round(float(item), 4) for item in epo_tr_batch_loss]

    epo_tr_batch_acc = utils.read_file("log/data/epo_tr_batch_acc.txt").split(",")
    epo_tr_batch_acc = [np.round(float(item), 4) for item in epo_tr_batch_acc]

    epo_te_batch_loss = utils.read_file("log/data/epo_te_batch_loss.txt").split(",")
    epo_te_batch_loss = [np.round(float(item), 4) for item in epo_te_batch_loss]

    epo_te_batch_acc = utils.read_file("log/data/epo_te_batch_acc.txt").split(",")
    epo_te_batch_acc = [np.round(float(item), 4) for item in epo_te_batch_acc]

    plot_loss_acc(epo_tr_batch_loss, epo_tr_batch_acc, "training")
    plot_loss_acc(epo_te_batch_loss, epo_te_batch_acc, "test")


def sample_balanced(x_seqs, y_labels, ulabels_tr_dict):
    batch_tools = list(ulabels_tr_dict.keys())
    random.shuffle(batch_tools)
    last_tools = batch_tools[:batch_size]
    rand_batch_indices = list()
    for l_tool in last_tools:
        seq_indices = ulabels_tr_dict[l_tool]
        random.shuffle(seq_indices)
        rand_batch_indices.append(seq_indices[0])

    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y


def get_u_labels(y_train):
    last_tools = list()
    ulabels_dict = dict()
    for item in range(y_train.shape[0]):
        arr_seq = y_train[item]
        #print(arr_seq)
        label_pos = np.where(arr_seq > 0)[0]
        #print(label_pos, arr_seq)
        last_tool = str(int(arr_seq[label_pos[-1]]))
        if last_tool not in ulabels_dict:
            ulabels_dict[last_tool] = list()
        ulabels_dict[last_tool].append(item)
        seq = ",".join([str(int(a)) for a in arr_seq[0:label_pos[-1] + 1]])
        #print(seq, last_tool)
        last_tools.append(last_tool)
        #print()
    u_labels = list(set(last_tools))
    #print(len(last_tools), len(u_labels))
    random.shuffle(u_labels)
    return u_labels, ulabels_dict


def predict_seq():
    n_topk = 10
    path_test_data = base_path + "saved_data/test.h5"
    file_obj = h5py.File(path_test_data, 'r')
    
    #test_target = tf.convert_to_tensor(np.array(file_obj["target"]), dtype=tf.int64)
    test_input = np.array(file_obj["input"])
    test_target = np.array(file_obj["target"])
    print(test_input.shape, test_target.shape)
    
    r_dict = utils.read_file(base_path + "data/rev_dict.txt")
    f_dict = utils.read_file(base_path + "data/f_dict.txt")

    class_weights = utils.read_file(base_path + "data/class_weights.txt")
    #print(class_weights)
    c_weights = list(class_weights.values())
    #print(len(c_weights))
    #c_weights.insert(0, 0.0)
    c_weights = tf.convert_to_tensor(c_weights, dtype=tf.float32)
    #print(c_weights.shape)
    
    tf_loaded_model = tf.saved_model.load(model_path)

    #u_te_labels, ulabels_te_dict  = get_u_labels(test_input)

    

    #te_x_batch, y_train_batch = sample_balanced(test_input, test_target, ulabels_te_dict)

    n_test = 1 #te_x_batch.shape[0]

    precision = list()
    for i in range(n_test):
        rand_index = i #np.random.randint(0, batch_size - 1)
        #print(rand_index)
        #t_ip = test_input[rand_index]
        #print(t_ip, test_target[rand_index])
        target_pos = [] #np.where(test_target[rand_index] > 0)[0]
        t_ip = np.zeros((25))
        t_ip[0] = int(f_dict["scanpy_find_markers"])
        #t_ip[1] = 637
        #print(t_ip)
        t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
        prediction, att_weights = tf_loaded_model([t_ip], training=False)
        prediction = tf.math.multiply(c_weights, prediction)
        #print(prediction.shape)
        
        #print(prediction.shape)
        if len(target_pos) < 5:
            #topk_pred = tf.math.top_k(te_pred_batch[idx], k=n_topk, sorted=True)
            top_k = tf.math.top_k(prediction, k=n_topk, sorted=True)
        else:
            top_k = tf.math.top_k(prediction, k=len(target_pos), sorted=True)
        
        #print("Top k: ", prediction.shape, top_k, top_k.indices)
        #print(np.all(top_k.indices.numpy(), axis=-1))
        t_ip = t_ip.numpy()

        label_pos = np.where(t_ip > 0)[0]
        
        #print(t_ip, t_ip[label_pos])
        #print(label_pos, arr_seq)
        i_names = ",".join([r_dict[str(item)] for item in t_ip[label_pos]  if item not in [0, "0"]])
        #t_names = ",".join([r_dict[str(int(item))] for item in target_pos  if item not in [0, "0"]])

        
        #print(i_names, top_k.indices.numpy()[0])
        
        true_tools = [r_dict[str(int(item))] for item in target_pos]
        pred_tools = [r_dict[str(item)] for item in top_k.indices.numpy()[0]  if item not in [0, "0"]]
        #intersection = list(set(true_tools).intersection(set(pred_tools)))
        pred_precision = 0.0 #float(len(intersection)) / len(true_tools)
        #precision.append(pred_precision)
        print("Tool sequence: {}".format([r_dict[str(item)] for item in t_ip[label_pos]]))
        print()
        print("True tools: {}".format(true_tools))
        print()
        print("Predicted top {} tools: {}".format(n_topk, pred_tools))
        print()
        print("Precision: {}".format(pred_precision))
        print("=========================")
        generated_attention(att_weights, i_names, f_dict, r_dict)
        if i == 100:
            break
    print("Precision@{}: {}".format(n_topk, np.mean(precision)))

def generated_attention(attention_weights, i_names, f_dict, r_dict):

    #print(attention_weights.shape)
    attention_heads = tf.squeeze(attention_weights, 0)
    i_names = i_names.split(",")
    in_tokens = i_names
    out_tokens = i_names
    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
      ax = fig.add_subplot(2, 4, h+1)
      plot_attention_head(in_tokens, out_tokens, head)
      ax.set_xlabel(f'Head {h+1}')
    plt.tight_layout()
    plt.show()


def plot_attention_head(in_tokens, out_tokens, attention):
  # The plot is of the attention when a token was generated.
  # The model didn't generate `<START>` in the output. Skip it.
  #translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention[:len(in_tokens), :len(out_tokens)])

  '''ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(out_tokens)))

  ax.set_xticklabels(in_tokens, rotation=90)
  ax.set_yticklabels(out_tokens)'''


'''
def predict_seq():


    #sys.exit()
    # read test sequences
    r_dict = utils.read_file(base_path + "data/rev_dict.txt")
    f_dict = utils.read_file(base_path + "data/f_dict.txt")
    
    tf_loaded_model = tf.saved_model.load(model_path)
    #predictor = predict_sequences.PredictSequence(tf_loaded_model)

    #predictor(test_input, test_target, f_dict, r_dict)

    #tool_name = "cutadapt"
    #print("Prediction for {}...".format(tool_name))
    bowtie_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    bowtie_output = bowtie_output.write(0, [tf.constant(index_start_token, dtype=tf.int64)])
    #bowtie_output = bowtie_output.write(1, [tf.constant(295, dtype=tf.int64)])
    bowtie_o = tf.transpose(bowtie_output.stack())
    #tool_id = f_dict[tool_name]
    #print(tool_name, tool_id)
    tool_list = ["ctb_filter"]
    bowtie_input = np.zeros([1, 25])
    bowtie_input[:, 0] = index_start_token
    bowtie_input[:, 1] = f_dict[tool_list[0]]
    #bowtie_input[:, 2] = f_dict[tool_list[1]]
    #bowtie_input[:, 3] = f_dict["featurecounts"]
    #bowtie_input[:, 4] = f_dict["deseq2"]
    bowtie_input = tf.constant(bowtie_input, dtype=tf.int64)
    print(bowtie_input, bowtie_output, bowtie_o)
    bowtie_pred, _ = tf_loaded_model([bowtie_input, bowtie_o], training=False)
    print(bowtie_pred.shape)
    top_k = tf.math.top_k(bowtie_pred, k=10)
    print("Top k: ", bowtie_pred.shape, top_k, top_k.indices)
    print(np.all(top_k.indices.numpy(), axis=-1))
    print("Predicted tools for {}: {}".format( ",".join(tool_list), [r_dict[str(item)] for item in top_k.indices.numpy()[0][0]]))
    print()
    #print("Generating predictions...")
    #generated_attention(tf_loaded_model, f_dict, r_dict)


def generated_attention(trained_model, f_dict, r_dict):

    np_output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    np_output_array = np_output_array.write(0, [tf.constant(index_start_token, dtype=tf.int64)])

    n_target_items = 5
    n_input = np.zeros([1, 25])
    n_input[:, 0] = index_start_token
    n_input[:, 1] = f_dict["hicexplorer_hicadjustmatrix"]
    #n_input[:, 2] = f_dict["hicexplorer_hicbuildmatrix"]
    #n_input[:, 3] = f_dict["hicexplorer_hicfindtads"]
    #n_input[:, 4] = f_dict["deseq2"]
    #n_input[:, 5] = f_dict["Add_a_column1"]
    #n_input[:, 6] = f_dict["table_compute"]
    a_input = n_input
    n_input = tf.constant(n_input, dtype=tf.int64)
   
    for i in range(n_target_items):
        #print(i, index)
        output = tf.transpose(np_output_array.stack())
        print("decoder input: ", n_input, output, output.shape)
        orig_predictions, _ = trained_model([n_input, output], training=False)
        #print(orig_predictions.shape)
        #print("true target seq real: ", te_tar_real)
        #print("Pred seq argmax: ", tf.argmax(orig_predictions, axis=-1))
        predictions = orig_predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        np_output_array = np_output_array.write(i+1, predicted_id[0])
    print(output, np_output_array.stack(), output.numpy())
    print("----------")
    last_decoder_layer = "decoder_layer4_block2"
    _, attention_weights = trained_model([n_input, output[:,:-1]], training=False)
    pred_attention = attention_weights[last_decoder_layer]

    print(attention_weights[last_decoder_layer].shape)
    head = 0
    attention_heads = tf.squeeze(attention_weights[last_decoder_layer], 0)
    pred_attention = attention_heads[head]
    print(pred_attention)

    #print(attention_weights)
    in_tokens = [r_dict[str(int(item))] for item in a_input[0] if item > 0]
    out_tokens = [r_dict[str(int(item))] for item in output.numpy()[0]]
    out_tokens = out_tokens[1:]
    print(in_tokens)
    print(out_tokens)
    pred_attention = pred_attention[:,:len(in_tokens)]
    print(pred_attention)
    plot_attention_head(in_tokens, out_tokens, pred_attention)


def plot_attention_head(in_tokens, out_tokens, attention):
  # The plot is of the attention when a token was generated.
  # The model didn't generate `<START>` in the output. Skip it.

  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attention, interpolation='nearest')
  fig.colorbar(cax)

  #ax = plt.gca()
  #ax.matshow(attention)

  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(out_tokens)))

  ax.set_xticklabels(in_tokens, rotation=90)
  ax.set_yticklabels(out_tokens)

  plt.show()

'''

if __name__ == "__main__":
    predict_seq()
    #visualize_loss_acc()
