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

'''embed_dim = 128 # Embedding size for each token
num_heads = 8 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer
d_dim = 128
dropout = 0.1
n_train_batches = 1000000

test_logging_step = 100
train_logging_step = 2000
n_test_seqs = batch_size
learning_rate = 1e-2'''

fig_size = (12, 12)
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

batch_size = 32
test_batches = 0
n_topk = 1

'''loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')'''


base_path = "log/"
predict_rnn = False
#model_path = base_path + "saved_model/382000/tf_model/"
model_number = 100000
model_path = base_path + "saved_model/" + str(model_number) + "/tf_model/"

'''
 ['dropletutils_read_10x', 'scmap_preprocess_sce']
['msnbase-read-msms', 'map-msms2camera', 'msms2metfrag-multiple', 'metfrag-cli-batch-multiple', 'passatutto']


'''


def verify_training_sampling(sampled_tool_ids, rev_dict):
    """
    Compute the frequency of tool sequences after oversampling
    """
    freq_dict = dict()
    freq_dict_names = dict()
    sampled_tool_ids = sampled_tool_ids.split(",")
    for tr_tool_id in sampled_tool_ids:
        tr_tool_id = tr_tool_id  #str(int(tr_data[-1]))
        if tr_tool_id not in freq_dict:
            freq_dict[tr_tool_id] = 0
            freq_dict_names[rev_dict[str(tr_tool_id)]] = 0
        freq_dict[tr_tool_id] += 1
        freq_dict_names[rev_dict[str(tr_tool_id)]] += 1
    #print(dict(sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)))
    s_freq = dict(sorted(freq_dict_names.items(), key=lambda kv: kv[1], reverse=True))
    print(s_freq)
    return s_freq



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
    epo_tr_batch_loss = utils.read_file(base_path + "data/epo_tr_batch_loss.txt").split(",")

    #print(len(epo_tr_batch_loss))
    epo_tr_batch_loss = [np.round(float(item), 4) for item in epo_tr_batch_loss]

    epo_tr_batch_acc = utils.read_file(base_path + "data/epo_tr_batch_acc.txt").split(",")
    epo_tr_batch_acc = [np.round(float(item), 4) for item in epo_tr_batch_acc]

    epo_te_batch_loss = utils.read_file(base_path + "data/epo_te_batch_loss.txt").split(",")
    epo_te_batch_loss = [np.round(float(item), 4) for item in epo_te_batch_loss]

    epo_te_batch_acc = utils.read_file(base_path + "data/epo_te_batch_acc.txt").split(",")
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


def get_u_tr_labels(y_tr):
    labels = list()
    labels_pos_dict = dict()
    for i, item in enumerate(y_tr):
        label_pos = np.where(item > 0)[0]
        labels.extend(label_pos)
        for label in label_pos:
            if label not in labels_pos_dict:
                labels_pos_dict[label] = list()
            labels_pos_dict[label].append(i)

    u_labels = list(set(labels))
    
    for item in labels_pos_dict:
        labels_pos_dict[item] = list(set(labels_pos_dict[item]))
    return u_labels, labels_pos_dict


'''

#TODO: to change

def get_u_tr_labels(x_tr, y_tr):
    labels = list()
    labels_pos_dict = dict()
    for i, (item_x, item_y) in enumerate(zip(x_tr, y_tr)):
        all_pos = list()
        label_pos = np.where(item_y > 0)[0]
        data_pos = np.where(item_x > 0)[0]
        data_pos = [int(a) for a in item_x[data_pos]]
        labels.extend(label_pos)
        labels.extend(data_pos) 
        all_pos.extend(label_pos)
        all_pos.extend(data_pos)
        #print(i, item_x, data_pos, label_pos)
        #print()
        for label in all_pos:
            if label not in labels_pos_dict:
                labels_pos_dict[label] = list()
            labels_pos_dict[label].append(i)

    u_labels = list(set(labels))
    
    for item in labels_pos_dict:
        labels_pos_dict[item] = list(set(labels_pos_dict[item]))
    #print(labels_pos_dict)
    #sys.exit()
    return u_labels, labels_pos_dict
'''


def sample_balanced_tr_y(x_seqs, y_labels, ulabels_tr_y_dict):
    batch_y_tools = list(ulabels_tr_y_dict.keys())
    random.shuffle(batch_y_tools)
    label_tools = list()
    rand_batch_indices = list()

    for l_tool in batch_y_tools:
        seq_indices = ulabels_tr_y_dict[l_tool]
        random.shuffle(seq_indices)
        
        if seq_indices[0] not in rand_batch_indices:
            rand_batch_indices.append(seq_indices[0])
            label_tools.append(l_tool)
        if len(rand_batch_indices) == batch_size:
            break
    
    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]

    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y, label_tools, rand_batch_indices


def predict_seq():

    #visualize_loss_acc()

    r_dict = utils.read_file(base_path + "data/rev_dict.txt")
    tool_tr_freq = utils.read_file(base_path + "data/all_sel_tool_ids.txt")
    verify_training_sampling(tool_tr_freq, r_dict)

    sys.exit()

    

    
    path_test_data = base_path + "saved_data/test.h5"
    file_obj = h5py.File(path_test_data, 'r')
    
    #test_target = tf.convert_to_tensor(np.array(file_obj["target"]), dtype=tf.int64)
    test_input = np.array(file_obj["input"])
    test_target = np.array(file_obj["target"])
    print(test_input.shape, test_target.shape)
    
    r_dict = utils.read_file(base_path + "data/rev_dict.txt")
    f_dict = utils.read_file(base_path + "data/f_dict.txt")

    class_weights = utils.read_file(base_path + "data/class_weights.txt")
    compatible_tools = utils.read_file(base_path + "data/compatible_tools.txt")

    tool_freq = utils.read_file(base_path + "data/freq_dict_names.txt")
    published_connections = utils.read_file(base_path + "data/published_connections.txt")

    

    c_weights = list(class_weights.values())

    c_weights = tf.convert_to_tensor(c_weights, dtype=tf.float32)

    tf_loaded_model = tf.saved_model.load(model_path)

    #u_te_labels, ulabels_te_dict  = get_u_labels(test_input)

    #u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(test_input, test_target)
    u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(test_target)

    precision = list()
    pub_prec_list = list()
    error_label_tools = list()
    for j in range(test_batches):
        #te_x_batch, y_train_batch = sample_balanced(test_input, test_target, ulabels_te_dict)
        te_x_batch, y_train_batch, selected_label_tools, bat_ind = sample_balanced_tr_y(test_input, test_target, u_te_y_labels_dict)
        print()
        print(bat_ind)
        print()
        print(selected_label_tools)
        print()
        select_tools = [r_dict[str(item)] for item in selected_label_tools]
        print("Selected label tools: {}".format(select_tools))
        #sys.exit()
        n_test = te_x_batch.shape[0]
        for i, (inp, tar) in enumerate(zip(te_x_batch, y_train_batch)):
            print("Test batch selected tool and index: {}, {}".format(bat_ind[i], select_tools[i]))
            s_label_tool = select_tools[i]
            t_ip = inp #test_input[i]
            target_pos = np.where(tar > 0)[0]
            t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
            if predict_rnn is True:
                prediction = tf_loaded_model([t_ip], training=False)
            else:
                prediction, att_weights = tf_loaded_model([t_ip], training=False)
            prediction_wts = tf.math.multiply(c_weights, prediction)

            #n_topk = len(target_pos)
            top_k = tf.math.top_k(prediction, k=n_topk, sorted=True)
            top_k_wts = tf.math.top_k(prediction_wts, k=n_topk, sorted=True)

            t_ip = t_ip.numpy()
            label_pos = np.where(t_ip > 0)[0]

            i_names = ",".join([r_dict[str(item)] for item in t_ip[label_pos]  if item not in [0, "0"]])
            t_names = ",".join([r_dict[str(int(item))] for item in target_pos  if item not in [0, "0"]])

            last_i_tool = [r_dict[str(item)] for item in t_ip[label_pos]][-1]

            true_tools = [r_dict[str(int(item))] for item in target_pos]

            print("Selected tool in true tools: {}".format(s_label_tool in true_tools))

            pred_tools = [r_dict[str(item)] for item in top_k.indices.numpy()[0]  if item not in [0, "0"]]
            pred_tools_wts = [r_dict[str(item)] for item in top_k_wts.indices.numpy()[0]  if item not in [0, "0"]]

            intersection = list(set(true_tools).intersection(set(pred_tools)))

            pub_prec = 0.0
            pub_prec_wt = 0.0

            if last_i_tool in published_connections:
                true_pub_conn = published_connections[last_i_tool]
                #print("Test batch {}, True published tools: {}".format(j+1, true_pub_conn)) 
                #print()
                intersection_pub = list(set(true_pub_conn).intersection(set(pred_tools)))
                intersection_pub_wt = list(set(true_pub_conn).intersection(set(pred_tools_wts)))
                pub_prec = float(len(intersection_pub)) / len(pred_tools)
                pub_prec_list.append(pub_prec)
                pub_prec_wt = float(len(intersection_pub_wt)) / len(pred_tools)
            else:
                pub_prec = False
                pub_prec_wt = False
            pred_precision = float(len(intersection)) / len(pred_tools)
            precision.append(pred_precision)

            if pred_precision < 1.0:
            
                print("Test batch {}, Tool sequence: {}".format(j+1, [r_dict[str(item)] for item in t_ip[label_pos]]))
                print()
                print("Test batch {}, True tools: {}".format(j+1, true_tools))
                print()
                print("Test batch {}, Predicted top {} tools: {}".format(j+1, n_topk, pred_tools))
                print()
                print("Test batch {}, Predicted top {} tools with weights: {}".format(j+1, n_topk, pred_tools_wts))
                print()
                print("Test batch {}, Precision: {}".format(j+1, pred_precision)) 
                print()
                print("Test batch {}, Published precision: {}".format(j+1, pub_prec))
                print()
                print("Test batch {}, Published precision with weights: {}".format(j+1, pub_prec_wt))
                error_label_tools.append(select_tools[i])
                print("=========================")
            print("--------------------------")
            #generated_attention(att_weights, i_names, f_dict, r_dict)
        print("Batch {} prediction finished ...".format(j+1))
    #print("Precision@{}: {}".format(n_topk, np.mean(precision)))
    #print("Published Precision@{}: {}".format(n_topk, np.mean(pub_prec_list)))
    #print("Low precision on labels: {}".format(error_label_tools))
    #print("Low precision on labels: {}, # tools: {}".format(list(set(error_label_tools)), len(list(set(error_label_tools)))))
    # individual tools or seq prediction
    print()
    n_topk_ind = 20
    print("Predicting for individual tools or sequences")
    t_ip = np.zeros((25))
    t_ip[0] = int(f_dict["bowtie2"])
    t_ip[1] = int(f_dict["hicexplorer_hicbuildmatrix"])
    t_ip[2] = int(f_dict["hicexplorer_hicfindtads"])
    #t_ip[3] = int(f_dict["pangolin"])
    # Tested tools: porechop, schicexplorer_schicqualitycontrol, schicexplorer_schicclustersvl, snpeff_sars_cov_2
    # sarscov2genomes, ivar_covid_aries_consensus, remove_nucleotide_deletions, pangolin
    # bowtie2,lofreq_call
    # dropletutils_read_10x
    # 'bowtie2', 'hicexplorer_hicbuildmatrix'
    
    last_tool_name = "hicexplorer_hicfindtads"
    t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
    if predict_rnn is True:
        prediction = tf_loaded_model([t_ip], training=False)
    else:
        prediction, att_weights = tf_loaded_model([t_ip], training=False)
    prediction_cwts = tf.math.multiply(c_weights, prediction)

    top_k = tf.math.top_k(prediction, k=n_topk_ind, sorted=True)
    top_k_wts = tf.math.top_k(prediction_cwts, k=n_topk_ind, sorted=True)

    t_ip = t_ip.numpy()
    label_pos = np.where(t_ip > 0)[0]

    i_names = ",".join([r_dict[str(item)] for item in t_ip[label_pos]  if item not in [0, "0"]])
    
    pred_tools = [r_dict[str(item)] for item in top_k.indices.numpy()[0]  if item not in [0, "0"]]
    pred_tools_wts = [r_dict[str(item)] for item in top_k_wts.indices.numpy()[0]  if item not in [0, "0"]]
   
    c_tools = [r_dict[str(item)] for item in compatible_tools[str(f_dict[last_tool_name])]]

    print("Tool sequence: {}".format([r_dict[str(item)] for item in t_ip[label_pos]]))
    print()
    print("Compatible true tools: {}, size: {}".format(c_tools, len(c_tools)))
    print()
    print("Predicted top {} tools: {}".format(n_topk_ind, pred_tools))
    print()
    print("Predicted top {} tools with weights: {}".format(n_topk_ind, pred_tools_wts))


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
