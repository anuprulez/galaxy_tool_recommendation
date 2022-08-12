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

fig_size = (15, 15)
font = {'family': 'serif', 'size': 8}
plt.rc('font', **font)

batch_size = 100
test_batches = 10
n_topk = 1
max_seq_len = 25


predict_rnn = False

if predict_rnn is True:
    base_path = "log_08_08_22_rnn/"
else:
    base_path = "log_local_11_08_22_2/" #"log_08_08_22_2/"  log_12_08_22_2 log_local_11_08_22_1

#predict_rnn = True # set to True for RNN model

#base_path = "log_08_08_22_2/"
#predict_rnn = False # set to True for RNN model

# log_08_08_22_2 (finish time: 40,000 steps in 158683.60082054138 seconds)
# log_08_08_22_rnn (finish time: 40,000 steps in 173480.83078551292 seconds)

#"log_03_08_22_1/" Balanced data with really selection of low freq tools - random choice
# RNN: log_01_08_22_3_rnn
# Transformer: log_01_08_22_0

model_number = 3900
onnx_model_path = base_path + "saved_model/"
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
        if tr_tool_id not in freq_dict:
            freq_dict[tr_tool_id] = 0
            freq_dict_names[rev_dict[str(tr_tool_id)]] = 0
        freq_dict[tr_tool_id] += 1
        freq_dict_names[rev_dict[str(tr_tool_id)]] += 1
    #print(dict(sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)))
    s_freq = dict(sorted(freq_dict_names.items(), key=lambda kv: kv[1], reverse=True))
    print(s_freq, len(s_freq))
    return s_freq


def plot_model_usage_time():
    steps = [1, 5, 10, 15, 20]

    rnn_time = [9.10, 8.229, 14.79, 10.39, 13.96]
    transformer_time = [0.92, 2.96, 0.97, 2.02, 1.68] 

    x_val = np.arange(len(steps))
    plt.plot(x_val, rnn_time)
    plt.plot(x_val, transformer_time)
    plt.ylabel("Model usage (load + prediction) time (seconds)")
    #plt.ylim((0.00, 0.07))
    plt.xlabel("Top K predictions")
    plt.xticks(x_val, [str(item) for item in steps])
    plt.legend(["RNN (GRU)", "Transformer"])
    plt.grid(True)
    plt.title("Model usage time comparison between RNN and Transformer".format())
    plt.savefig(base_path + "data/model_usage_time_rnn_transformer.png", dpi=150)
    plt.show()


def plot_rnn_transformer(tr_loss, te_loss):
    # plot training loss
    tr_pos_plot = [5000, 10000, 20000, 30000, 40000, 75000, 100000]
    te_pos_plot = [50, 100, 200, 300, 400, 750, 1000]

    print(len(tr_loss), len(te_loss))

    tr_loss_val = [tr_loss[item] for item in tr_pos_plot]
    te_loss_val = [te_loss[item] for item in te_pos_plot]

    print(tr_loss_val)
    print(te_loss_val)

    x_val = np.arange(len(tr_loss_val))
    plt.plot(x_val, tr_loss_val)
    plt.plot(x_val, te_loss_val)
    plt.ylabel("Loss")
    plt.ylim((0.00, 0.07))
    plt.xlabel("Training steps")
    plt.xticks(x_val, [str(item) for item in tr_pos_plot])
    plt.legend(["Training", "Test"])
    plt.grid(True)
    plt.title("Transformer training and test loss".format())
    plt.savefig(base_path + "data/{}_loss.png".format("transformer_tr_te_loss"), dpi=150)
    plt.show()

    # tr_pos_plot = [5000, 10000, 20000, 30000, 40000, 75000, 100000]
    rnn_te_prec = [0.22, 0.42, 0.68, 0.84, 0.89, 0.95, 0.9579]
    transformer_te_prec = [0.79, 0.90, 0.93, 0.94, 0.948, 0.953, 0.950]

    # plot topk precision for RNN and Transformer
    '''x_val = np.arange(len(rnn_te_prec))
    #x_val = np.arange(n_epo)
    plt.plot(x_val, rnn_te_prec)
    plt.plot(x_val, transformer_te_prec)
    plt.ylabel("Precision@k")
    plt.xlabel("Training steps")
    plt.xticks(x_val, [str(item) for item in tr_pos_plot])
    plt.legend(["RNN (GRU)", "Transformer"])
    plt.grid(True)
    plt.title("(Test) Precision@k for RNN (GRU) (~20 hrs) and Transformer (~ 12 hrs)")
    plt.savefig(base_path + "data/precision_k_rnn_vs_transformer.png", dpi=150)
    plt.show()'''


def plot_loss_acc(loss, acc, t_value):
    # plot training loss
    x_val = np.arange(len(loss))
    '''if t_value == "test":
        x_val = 20 * x_val'''
    plt.plot(x_val, loss)
    plt.ylabel("Loss".format(t_value))
    plt.xlabel("Training steps")
    plt.grid(True)
    plt.title("{} vs loss".format(t_value))
    plt.savefig(base_path + "/data/{}_loss.pdf".format(t_value), dpi=200)
    plt.show()

    # plot driver gene precision vs epochs
    x_val_acc = np.arange(len(acc))
    #x_val = np.arange(n_epo)
    plt.plot(x_val_acc, acc)
    plt.ylabel("Accuracy".format(t_value))
    plt.xlabel("Training steps")
    plt.grid(True)
    plt.title("{} steps vs accuracy".format(t_value))
    plt.savefig(base_path + "/data/{}_acc.pdf".format(t_value), dpi=200)
    plt.show()


def plot_low_te_prec(prec, t_value):
    x_val_acc = np.arange(len(prec))
    plt.plot(x_val_acc, prec)
    plt.ylabel("Precision@k".format(t_value))
    plt.xlabel("Training steps")
    plt.grid(True)
    plt.title("{} steps vs accuracy".format(t_value))
    plt.savefig(base_path + "/data/{}_low_acc.pdf".format(t_value), dpi=200)
    plt.show()


def visualize_loss_acc():
    epo_tr_batch_loss = utils.read_file(base_path + "data/epo_tr_batch_loss.txt").split(",")

    #print(len(epo_tr_batch_loss))
    epo_tr_batch_loss = [np.round(float(item), 4) for item in epo_tr_batch_loss]

    epo_tr_batch_acc = utils.read_file(base_path + "data/epo_tr_batch_acc.txt").split(",")
    epo_tr_batch_acc = [np.round(float(item), 4) for item in epo_tr_batch_acc]

    epo_te_batch_loss = utils.read_file(base_path + "data/epo_te_batch_loss.txt").split(",")
    epo_te_batch_loss = [np.round(float(item), 4) for item in epo_te_batch_loss]

    epo_te_batch_acc = utils.read_file(base_path + "data/epo_te_precision.txt").split(",")
    epo_te_batch_acc = [np.round(float(item), 4) for item in epo_te_batch_acc]

    plot_loss_acc(epo_tr_batch_loss, epo_tr_batch_acc, "training")
    plot_loss_acc(epo_te_batch_loss, epo_te_batch_acc, "test")

    epo_te_low_batch_acc = utils.read_file(base_path + "data/epo_low_te_precision.txt").split(",")
    epo_te_low_batch_acc = [np.round(float(item), 4) for item in epo_te_low_batch_acc]

    plot_low_te_prec(epo_te_low_batch_acc, "Low test")
    
    #plot_rnn_transformer(epo_tr_batch_loss, epo_te_batch_loss)


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


def verify_tool_in_tr(r_dict):
    all_sel_tool_ids = utils.read_file(base_path + "data/all_sel_tool_ids.txt").split(",")

    freq_dict = dict()
    freq_dict_names = dict()

    for tool_id in all_sel_tool_ids:
        if tool_id not in freq_dict:
            freq_dict[tool_id] = 0

        if tool_id not in freq_dict_names:
            freq_dict_names[r_dict[str(int(tool_id))]] = 0

        freq_dict[tool_id] += 1
        freq_dict_names[r_dict[str(int(tool_id))]] += 1

    s_freq = dict(sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True))
    s_freq_names = dict(sorted(freq_dict_names.items(), key=lambda kv: kv[1], reverse=True))

    utils.write_file(base_path + "data/s_freq_names.txt", s_freq_names)
    utils.write_file(base_path + "data/s_freq.txt", s_freq)

    return s_freq


def predict_seq():

    #visualize_loss_acc()

    #plot_model_usage_time()

    r_dict = utils.read_file(base_path + "data/rev_dict.txt")
    #tool_tr_freq = utils.read_file(base_path + "data/all_sel_tool_ids.txt")
    
    #verify_training_sampling(tool_tr_freq, r_dict)
    
    #all_tr_label_tools = verify_tool_in_tr(r_dict)

    #all_tr_label_tools_ids = list(all_tr_label_tools.keys())
    #all_tr_label_tools_ids = [int(t) for t in all_tr_label_tools_ids]
    
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

    #tool_freq = utils.read_file(base_path + "data/freq_dict_names.txt")
    published_connections = utils.read_file(base_path + "data/published_connections.txt")

    c_weights = list(class_weights.values())

    c_weights = tf.convert_to_tensor(c_weights, dtype=tf.float32)

    m_load_s_time = time.time()
    tf_loaded_model = tf.saved_model.load(model_path)
    m_load_e_time = time.time()
    model_loading_time = m_load_e_time - m_load_s_time

    '''print("Saving as ONNX model...")
    onnx_model_save_path = onnx_model_path + "{}/onnx_model/".format(model_number)
    utils.convert_to_onnx(model_path, onnx_model_save_path)'''

    u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(test_target)

    precision = list()
    pub_prec_list = list()
    error_label_tools = list()
    #seq_len = 5
    #test_batches = 10000
    batch_pred_time = list()
    for j in range(test_batches):

        te_x_batch, y_train_batch, selected_label_tools, bat_ind = sample_balanced_tr_y(test_input, test_target, u_te_y_labels_dict)
        #print(j * batch_size, j * batch_size + batch_size)
        #te_x_batch = test_input[j * batch_size : j * batch_size + batch_size, :]
        #y_train_batch = test_target[j * batch_size : j * batch_size + batch_size, :]
        
        for i, (inp, tar) in enumerate(zip(te_x_batch, y_train_batch)):

            t_ip = inp
            if len(np.where(inp > 0)[0]) <= max_seq_len:
                real_prediction = np.where(tar > 0)[0]
                target_pos = real_prediction #list(set(all_tr_label_tools_ids).intersection(set(real_prediction)))

                t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
                pred_s_time = time.time()
                if predict_rnn is True:
                    prediction = tf_loaded_model([t_ip], training=False)
                else:
                    prediction, att_weights = tf_loaded_model([t_ip], training=False)
                pred_e_time = time.time()
                diff_time = pred_e_time - pred_s_time
                batch_pred_time.append(diff_time)
                prediction_wts = tf.math.multiply(c_weights, prediction)

                n_topk = len(target_pos)
                top_k = tf.math.top_k(prediction, k=n_topk, sorted=True)
                top_k_wts = tf.math.top_k(prediction_wts, k=n_topk, sorted=True)

                t_ip = t_ip.numpy()
                label_pos = np.where(t_ip > 0)[0]

                i_names = ",".join([r_dict[str(item)] for item in t_ip[label_pos]  if item not in [0, "0"]])
                t_names = ",".join([r_dict[str(int(item))] for item in target_pos  if item not in [0, "0"]])

                last_i_tool = [r_dict[str(item)] for item in t_ip[label_pos]][-1]

                true_tools = [r_dict[str(int(item))] for item in target_pos]

                pred_tools = [r_dict[str(item)] for item in top_k.indices.numpy()[0]  if item not in [0, "0"]]
                pred_tools_wts = [r_dict[str(item)] for item in top_k_wts.indices.numpy()[0]  if item not in [0, "0"]]

                intersection = list(set(true_tools).intersection(set(pred_tools)))

                pub_prec = 0.0
                pub_prec_wt = 0.0

                if last_i_tool in published_connections:
                    true_pub_conn = published_connections[last_i_tool]

                    if len(pred_tools) > 0:
                        intersection_pub = list(set(true_pub_conn).intersection(set(pred_tools)))
                        intersection_pub_wt = list(set(true_pub_conn).intersection(set(pred_tools_wts)))
                        pub_prec = float(len(intersection_pub)) / len(pred_tools)
                        pub_prec_list.append(pub_prec)
                        pub_prec_wt = float(len(intersection_pub_wt)) / len(pred_tools)
                    else:
                        pub_prec = False
                        pub_prec_wt = False

                if len(pred_tools) > 0:
                    pred_precision = float(len(intersection)) / len(pred_tools)
                    precision.append(pred_precision)

                if pred_precision < 2.0:
            
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
                    print()
                    print("Time taken to predict tools: {} seconds".format(diff_time))
                    #error_label_tools.append(select_tools[i])
                    print("=========================")
                print("--------------------------")
                generated_attention(att_weights, i_names, f_dict, r_dict)
        
                print("Batch {} prediction finished ...".format(j+1))

    te_lowest_t_ids = utils.read_file(base_path + "data/te_lowest_t_ids.txt")
    lowest_t_ids = [int(item) for item in te_lowest_t_ids.split(",")]
    
    low_te_data = test_input[lowest_t_ids]
    low_te_labels = test_target[lowest_t_ids]
    #print("Test lowest ids", low_te_data.shape, low_te_labels.shape)
    #low_te_pred_batch, low_att_weights = tf_loaded_model([low_te_data], training=False)
    low_topk = 20
    low_te_precision = list()
    low_te_pred_time = list()
    for i, (low_inp, low_tar) in enumerate(zip(low_te_data, low_te_labels)):
        pred_s_time = time.time()
        if predict_rnn is True:
            low_prediction = tf_loaded_model([low_inp], training=False)
        else:
            low_prediction, att_weights = tf_loaded_model([low_inp], training=False)
        pred_e_time = time.time()
        low_diff_pred_t = pred_e_time - pred_s_time
        low_te_pred_time.append(low_diff_pred_t)
        print("Time taken to predict tools: {} seconds".format(low_diff_pred_t))
        low_label_pos = np.where(low_tar > 0)[0]
        low_topk = len(low_label_pos)
        low_topk_pred = tf.math.top_k(low_prediction, k=low_topk, sorted=True)
        low_topk_pred = low_topk_pred.indices.numpy()[0]
        
        low_label_pos_tools = [r_dict[str(item)] for item in low_label_pos if item not in [0, "0"]]
        low_pred_label_pos_tools = [r_dict[str(item)] for item in low_topk_pred if item not in [0, "0"]]

        low_intersection = list(set(low_label_pos_tools).intersection(set(low_pred_label_pos_tools)))
        low_pred_precision = float(len(low_intersection)) / len(low_label_pos)
        low_te_precision.append(low_pred_precision)

        low_inp_pos = np.where(low_inp > 0)[0]

        print("{}, Low: test tool sequence: {}".format(i, [r_dict[str(int(item))] for item in low_inp[low_inp_pos]]))
        print()
        print("{},Low: True labels: {}".format(i, low_label_pos_tools))
        print()
        print("{},Low: Predicted labels: {}, Precision: {}".format(i, low_pred_label_pos_tools, low_pred_precision))
       
        print("-----------------")
        print()

    if test_batches > 0:
        print("Batch Precision@{}: {}".format(n_topk, np.mean(precision)))
        print("Batch Published Precision@{}: {}".format(n_topk, np.mean(pub_prec_list)))
        print("Batch Trained model loading time: {} seconds".format(model_loading_time))
        print("Batch average seq pred time: {} seconds".format(np.mean(batch_pred_time)))
        print("Batch total model loading and pred time: {} seconds".format(model_loading_time + np.mean(batch_pred_time)))
        print()

    if len(lowest_t_ids) > 0:
        print("Low test prediction precision: {}".format(np.mean(low_te_precision)))
        print()
        print("Low: test average prediction time: {}".format(np.mean(low_te_pred_time)))
        print()
    #print("Low precision on labels: {}".format(error_label_tools))
    #print("Low precision on labels: {}, # tools: {}".format(list(set(error_label_tools)), len(list(set(error_label_tools)))))
    # individual tools or seq prediction
    '''print()
    n_topk_ind = 20
    print("Predicting for individual tools or sequences")
    t_ip = np.zeros((25))
    t_ip[0] = int(f_dict["ncbi_eutils_esearch"])
    #t_ip[1] = int(f_dict["mtbls520_05a_import_maf"])
    #t_ip[2] = int(f_dict["mtbls520_06_import_traits"])
    #t_ip[3] = int(f_dict["mtbls520_07_species_diversity"])'''
    n_topk_ind = 20
    t_ip = np.zeros((25))
    '''t_ip[0] = int(f_dict["bowtie2"])
    t_ip[1] = int(f_dict["hicexplorer_hicbuildmatrix"])
    t_ip[2] = int(f_dict["hicexplorer_hicfindtads"])
    t_ip[3] = int(f_dict["hicexplorer_hicpca"])'''

    t_ip[0] = int(f_dict["deseq2"])
    t_ip[1] = int(f_dict["tp_cut_tool"])
    t_ip[2] = int(f_dict["heinz_scoring"])
    t_ip[3] = int(f_dict["heinz"])
    #t_ip[4] = int(f_dict["prokka"])
    #t_ip[5] = int(f_dict["roary"]) 
    #t_ip[6] = int(f_dict["Cut1"])
    #t_ip[7] = int(f_dict["cat1"])
    #t_ip[8] = int(f_dict["anndata_manipulate"])
    # 'snpEff_build_gb', 'bwa_mem', 'samtools_view',
    last_tool_name = "heinz"
    
    t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
    pred_s_time = time.time()
    if predict_rnn is True:
        prediction = tf_loaded_model([t_ip], training=False)
    else:
        prediction, att_weights = tf_loaded_model([t_ip], training=False)
    pred_e_time = time.time()
    print("Time taken to predict tools: {} seconds".format(pred_e_time - pred_s_time))
    prediction_cwts = tf.math.multiply(c_weights, prediction)

    top_k = tf.math.top_k(prediction, k=n_topk_ind, sorted=True)
    top_k_wts = tf.math.top_k(prediction_cwts, k=n_topk_ind, sorted=True)

    t_ip = t_ip.numpy()
    label_pos = np.where(t_ip > 0)[0]

    i_names = ",".join([r_dict[str(item)] for item in t_ip[label_pos]  if item not in [0, "0"]])
    
    pred_tools = [r_dict[str(item)] for item in top_k.indices.numpy()[0]  if item not in [0, "0"]]
    pred_tools_wts = [r_dict[str(item)] for item in top_k_wts.indices.numpy()[0]  if item not in [0, "0"]]
   
    c_tools = []
    if str(f_dict[last_tool_name]) in compatible_tools:
        c_tools = [r_dict[str(item)] for item in compatible_tools[str(f_dict[last_tool_name])]]

    pred_intersection = list(set(pred_tools).intersection(set(c_tools)))
    prd_te_prec = len(pred_intersection) / float(n_topk_ind)

    print("Tool sequence: {}".format([r_dict[str(item)] for item in t_ip[label_pos]]))
    print()
    print("Compatible true tools: {}, size: {}".format(c_tools, len(c_tools)))
    print()
    print("Predicted top {} tools: {}".format(n_topk_ind, pred_tools))
    print()
    print("Predicted precision: {}".format(prd_te_prec))
    print()
    print("Correctly predicted tools: {}".format(pred_intersection))
    print()
    print("Predicted top {} tools with weights: {}".format(n_topk_ind, pred_tools_wts))
    print()
    if predict_rnn is False:
        generated_attention(att_weights, i_names, f_dict, r_dict)


def generated_attention(attention_weights, i_names, f_dict, r_dict):

    print(attention_weights.shape)
    attention_heads = tf.squeeze(attention_weights, 0)
    n_heads = attention_heads.shape[1]
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
  #print(attention)
  ax = plt.gca()
  ax.matshow(attention[:len(in_tokens), :len(out_tokens)])
  #ax.matshow(attention)

  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(out_tokens)))

  ax.set_xticklabels(in_tokens, rotation=90)
  ax.set_yticklabels(out_tokens)

'''

Tool seqs for good attention plots:
'schicexplorer_schicqualitycontrol', 'schicexplorer_schicnormalize', 'schicexplorer_schicclustersvl'


# Tested tools: porechop, schicexplorer_schicqualitycontrol, schicexplorer_schicclustersvl, snpeff_sars_cov_2
    # sarscov2genomes, ivar_covid_aries_consensus, remove_nucleotide_deletions, pangolin
    # bowtie2,lofreq_call
    # dropletutils_read_10x
    # 'bowtie2', 'hicexplorer_hicbuildmatrix'
    # 'mtbls520_04_preparations', 'mtbls520_05a_import_maf', 'mtbls520_06_import_traits', 'mtbls520_07_species_diversity'
    # ctsm_fates: 'xarray_metadata_info', 'interactive_tool_panoply', 'xarray_select', '__EXTRACT_DATASET__'
    # msnbase_readmsdata: 'abims_xcms_xcmsSet', 'xcms_export_samplemetadata', 'xcms_plot_chromatogram'
    # ncbi_eutils_esearch: ncbi_eutils_elink
    # 1_create_conf: '5_calc_stat', '4_filter_sam', '2_map', 'conf4circos', '3_filter_single_pair'
    # pdaug_peptide_data_access: pdaug_tsvtofasta
    # 'pdaug_peptide_data_access', 'pdaug_tsvtofasta': 'pdaug_peptide_sequence_analysis', 'pdaug_fishers_plot', 'pdaug_sequence_property_based_descriptors'
    # 'rankprodthree', 'Remove beginning1', 'cat1', 'Cut1', 'interactions': 'biotranslator', 'awkscript'
    # rpExtractSink: rpCompletion', 'retropath2'
    # 'EMBOSS: transeq101', 'ncbi_makeblastdb', 'ncbi_blastp_wrapper', 'blast_parser', 'hcluster_sg'
    # 'Remove beginning1', 'Cut1', 'param_value_from_file', 'kc-align', 'sarscov2formatter', 'hyphy_fel'
    # abims_CAMERA_annotateDiffreport
    # cooler_csort_pairix
    # mycrobiota-split-multi-otutable
    # XY_Plot_1
    # mycrobiota-qc-report
    # 1_create_conf
    # RNAlien
    # ont_fast5_api_multi_to_single_fast5 
    # ctb_remIons

    Incorrect predictions
    # scpipe, 
    # 'delly_call', 'delly_merge'
    # 'gmap_build', 'gsnap', 'sam_to_bam', 'filter', 'assign', 'polyA'
    # 'bioext_bealign', 'tn93_filter', 'hyphy_cfel'
    # sklearn_build_pipeline
    # split_file_to_collection', 'rdock_rbdock', 'xchem_pose_scoring', 'sucos_max_score'
    # 'rmcontamination', 'scaffold2fasta'  
    # 'rmcontamination', 'scaffold2fasta'
    # cat1', 'fastq_filter', 'cshl_fastq_to_fasta', 'filter_16s_wrapper_script 1'
    # 'TrimPrimer', 'Flash', 'Btrim64', 'uparse'
    # 'cshl_fastq_to_fasta', 'cshl_fastx_trimmer', 'fasta_tabular_converter
    # CryptoGenotyper
    # cooler_makebins
    # 'PeakPickerHiRes', 'FileFilter', 'xcms-find-peaks', 'xcms-collect-peaks'
    # 'TrimPrimer', 'Flash', 'Btrim64'
    # cryptotyper
    # ip_spot_detection_2d
    # 'picard_FastqToSam', 'TagBamWithReadSequenceExtended', 'FilterBAM', 'BAMTagHistogram'
    # 'basic_illumination', 'ashlar'
    # 'cghub_genetorrent', 'gatk_indel'
    # 'FeatureFinderMultiplex', 'HighResPrecursorMassCorrector', 'MSGFPlusAdapter', 'PeptideIndexer', 'IDMerger', 'ConsensusID'
    # 'PeakPickerHiRes', 'FileFilter', 'xcms-find-peaks', 'xcms-collect-peaks'
    # 'PeakPickerHiRes', 'FileFilter', 'xcms-find-peaks', 'xcms-collect-peaks', 'xcms-group-peaks', 'xcms-blankfilter', 'xcms-dilutionfilter', 'camera-annotate-peaks', 'camera-group-fwhm', 'camera-find-adducts', 'camera-find-isotopes'
    # 'minfi_read450k', 'minfi_mset'
    # 'msnbase_readmsdata', 'abims_xcms_xcmsSet', 'abims_xcms_refine'
    # # 'snpEff_build_gb', 'bwa_mem', 'samtools_view',

'''

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
