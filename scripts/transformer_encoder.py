import os
import time
import random
import numpy as np
import subprocess
import sys
import json

import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model

import utils

'''

Best model Transformer

embed_dim = 128 # Embedding size for each token d_model
num_heads = 4 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer # dff
#d_dim = 512
dropout = 0.1
n_train_batches = 40000
batch_size = 512
test_logging_step = 20
train_logging_step = 1000
te_batch_size = batch_size
learning_rate = 1e-3 #2e-5 #1e-3

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        # TODO: Add kernel_regularizer='l2', activity_regularizer='l2' to MHA
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"),
             #Dense(ff_dim, activation="relu"),
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output, attention_scores = self.att(inputs, inputs, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attention_scores


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def create_model(maxlen, vocab_size):
    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x, weights = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=[outputs, weights])

'''


embed_dim = 128 # Embedding size for each token d_model
num_heads = 4 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer # dff
#d_dim = 512
dropout = 0.1
n_train_batches = 40000
batch_size = 512
test_logging_step = 10
train_logging_step = 1000
te_batch_size = batch_size
learning_rate = 1e-3 #2e-5 #1e-3

# Readings
# https://keras.io/examples/nlp/text_classification_with_transformer/
# https://hannibunny.github.io/mlbook/transformer/attention.html
# https://arxiv.org/pdf/1905.09418.pdf
# https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15794705.pdf
# https://royalsocietypublishing.org/doi/10.1098/rsos.191517
# https://arxiv.org/pdf/1909.06639.pdf


#Train data:  (414952, 25)
#Test data:  (103739, 25)

binary_ce = tf.keras.losses.BinaryCrossentropy()
binary_fce = tf.keras.losses.BinaryFocalCrossentropy(gamma=5)
binary_acc = tf.keras.metrics.BinaryAccuracy()
categorical_ce = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        # TODO: Add kernel_regularizer='l2', activity_regularizer='l2' to MHA
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, a_mask, training):
        attn_output, attention_scores = self.att(inputs, inputs, inputs, attention_mask=a_mask, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attention_scores


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def create_model(maxlen, vocab_size):
    inputs = Input(shape=(maxlen,))
    a_mask = Input(shape=(1, maxlen))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x, weights = transformer_block(x, a_mask)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation="sigmoid")(x)
    return Model(inputs=[inputs, a_mask], outputs=[outputs, weights])


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


def sample_test_x_y(X, y):
    rand_batch_indices = np.random.randint(0, X.shape[0], batch_size)
    x_batch_train = X[rand_batch_indices]
    y_batch_train = y[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y


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


def sample_balanced_te_y(x_seqs, y_labels, ulabels_tr_y_dict, b_size):
    batch_y_tools = list(ulabels_tr_y_dict.keys())
    random.shuffle(batch_y_tools)
    label_tools = list()
    rand_batch_indices = list()
    sel_tools = list()
    for l_tool in batch_y_tools:
        seq_indices = ulabels_tr_y_dict[l_tool]
        random.shuffle(seq_indices)
        rand_s_index = np.random.randint(0, len(seq_indices), 1)[0]
        rand_sample = seq_indices[rand_s_index]
        sel_tools.append(l_tool)
        if rand_sample not in rand_batch_indices:
            rand_batch_indices.append(rand_sample)
            label_tools.append(l_tool)
        if len(rand_batch_indices) == b_size:
            break
    
    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]

    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y, sel_tools     


def sample_balanced_tr_y(x_seqs, y_labels, ulabels_tr_y_dict, b_size, tr_t_freq, prev_sel_tools):
    batch_y_tools = list(ulabels_tr_y_dict.keys())
    random.shuffle(batch_y_tools)
    label_tools = list()
    rand_batch_indices = list()
    sel_tools = list()

    unselected_tools = [t for t in batch_y_tools if t not in prev_sel_tools]
    rand_selected_tools = unselected_tools[:b_size]

    for l_tool in rand_selected_tools:
        seq_indices = ulabels_tr_y_dict[l_tool]
        random.shuffle(seq_indices)
        rand_s_index = np.random.randint(0, len(seq_indices), 1)[0]
        rand_sample = seq_indices[rand_s_index]
        sel_tools.append(l_tool)
        rand_batch_indices.append(rand_sample)
        label_tools.append(l_tool)

    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]

    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y, sel_tools


def compute_loss(y_true, y_pred, class_weights=None):
    y_true = tf.cast(y_true, dtype=tf.float32)
    loss = binary_ce(y_true, y_pred)
    #loss = binary_fce(y_true, y_pred)
    categorical_loss = categorical_ce(y_true, y_pred)

    if class_weights is None:
        return tf.reduce_mean(loss), categorical_loss
    return tf.tensordot(loss, class_weights, axes=1), categorical_loss


def compute_acc(y_true, y_pred):
    return binary_acc(y_true, y_pred)


def compute_topk_acc(y_true, y_pred, k):
    topk_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=k)
    return topk_acc



def validate_model(te_x, te_y, model, f_dict, r_dict, ulabels_te_dict, tr_labels, lowest_t_ids):
    #te_x_batch, y_train_batch = sample_test_x_y(te_x, te_y)
    #te_x_batch, y_train_batch = sample_balanced(te_x, te_y, ulabels_te_dict)
    print("Total test data size: ", te_x.shape, te_y.shape)
    te_x_batch, y_train_batch, _ = sample_balanced_te_y(te_x, te_y, ulabels_te_dict, te_batch_size)
    te_mask = utils.create_attention_mask(te_x_batch)
    print("Batch test data size: ", te_x_batch.shape, y_train_batch.shape)
    te_pred_batch, att_weights = model([te_x_batch, te_mask], training=False)
    test_acc = tf.reduce_mean(compute_acc(y_train_batch, te_pred_batch))
    test_err, test_categorical_loss = compute_loss(y_train_batch, te_pred_batch)

    te_pre_precision = list()
    
    for idx in range(te_pred_batch.shape[0]):
        label_pos = np.where(y_train_batch[idx] > 0)[0] 
        # verify only on those tools are present in labels in training
        label_pos = list(set(tr_labels).intersection(set(label_pos)))
        topk_pred = tf.math.top_k(te_pred_batch[idx], k=len(label_pos), sorted=True)
        topk_pred = topk_pred.indices.numpy()
        try:
            label_pos_tools = [r_dict[str(item)] for item in label_pos if item not in [0, "0"]]
            pred_label_pos_tools = [r_dict[str(item)] for item in topk_pred if item not in [0, "0"]]
        except:
            label_pos_tools = [r_dict[item] for item in label_pos if item not in [0, "0"]]
            pred_label_pos_tools = [r_dict[item] for item in topk_pred if item not in [0, "0"]]
        intersection = list(set(label_pos_tools).intersection(set(pred_label_pos_tools)))
        if len(topk_pred) > 0:
            pred_precision = float(len(intersection)) / len(topk_pred)
            te_pre_precision.append(pred_precision)
            print("True labels: {}".format(label_pos_tools))
            print()
            print("Predicted labels: {}, Precision: {}".format(pred_label_pos_tools, pred_precision))
            print("-----------------")
            print()

        if idx == te_batch_size - 1:
            break

    print("Test lowest ids", len(lowest_t_ids))
    low_te_data = te_x[lowest_t_ids]
    low_te_labels = te_y[lowest_t_ids]
    low_att_mask = utils.create_attention_mask(low_te_data)
    low_te_pred_batch, low_att_weights = model([low_te_data, low_att_mask], training=False)
    low_test_err, low_test_categorical_loss = compute_loss(low_te_labels, low_te_pred_batch)

    low_te_precision = list()
    for idx in range(low_te_pred_batch.shape[0]):
        low_label_pos = np.where(low_te_labels[idx] > 0)[0]
        low_label_pos = list(set(tr_labels).intersection(set(low_label_pos)))
        low_topk_pred = tf.math.top_k(low_te_pred_batch[idx], k=len(low_label_pos), sorted=True)
        low_topk_pred = low_topk_pred.indices.numpy()
        try:
            low_label_pos_tools = [r_dict[str(item)] for item in low_label_pos if item not in [0, "0"]]
            low_pred_label_pos_tools = [r_dict[str(item)] for item in low_topk_pred if item not in [0, "0"]]
        except:
            low_label_pos_tools = [r_dict[item] for item in low_label_pos if item not in [0, "0"]]
            low_pred_label_pos_tools = [r_dict[item] for item in low_topk_pred if item not in [0, "0"]]

        low_intersection = list(set(low_label_pos_tools).intersection(set(low_pred_label_pos_tools)))
        if len(low_label_pos) > 0:
            low_pred_precision = float(len(low_intersection)) / len(low_label_pos)
            low_te_precision.append(low_pred_precision)
            print("Low: True labels: {}".format(low_label_pos_tools))
            print()
            print("Low: Predicted labels: {}, Precision: {}".format(low_pred_label_pos_tools, low_pred_precision))
            print("-----------------")
            print()

    print("Test binary error: {}, test categorical loss: {}, test categorical accuracy: {}".format(test_err.numpy(), test_categorical_loss.numpy(), test_acc.numpy()))
    print("Test prediction precision: {}".format(np.mean(te_pre_precision)))
    print("Low test binary error: {}".format(low_test_err.numpy()))
    print("Low test prediction precision: {}".format(np.mean(low_te_precision)))

    print("Test finished")
    return test_err.numpy(), test_acc.numpy(), test_categorical_loss.numpy(), np.mean(te_pre_precision), np.mean(low_te_precision)


def create_enc_transformer(train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, tr_t_freq):
    print("Train transformer...")
    vocab_size = len(f_dict) + 1
    maxlen = train_data.shape[1]

    enc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = create_model(maxlen, vocab_size)

    u_tr_y_labels, u_tr_y_labels_dict = get_u_tr_labels(train_labels)
    u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(test_labels)

    trained_on_labels = [int(item) for item in list(u_tr_y_labels_dict.keys())]

    epo_tr_batch_loss = list()
    epo_tr_batch_acc = list()
    epo_tr_batch_categorical_loss = list()
    epo_te_batch_loss = list()
    epo_te_batch_acc = list()
    epo_te_batch_categorical_loss = list()
    all_sel_tool_ids = list()
    epo_te_precision = list()
    epo_low_te_precision = list()
    c_weights = tf.convert_to_tensor(list(c_wts.values()), dtype=tf.float32)

    te_lowest_t_ids = utils.get_low_freq_te_samples(test_data, test_labels, tr_t_freq)
    utils.write_file("log/data/te_lowest_t_ids.txt", ",".join([str(item) for item in te_lowest_t_ids]))

    sel_tools = list()
    for batch in range(n_train_batches):
        
        print("Total train data size: ", train_data.shape, train_labels.shape)
        #x_train, y_train = sample_test_x_y(train_data, train_labels)
        #x_train, y_train = sample_balanced(train_data, train_labels, ulabels_tr_dict)
        x_train, y_train, sel_tools = sample_balanced_tr_y(train_data, train_labels, u_tr_y_labels_dict, batch_size, tr_t_freq, sel_tools)
        att_mask = utils.create_attention_mask(x_train)

        print("Batch train data size: ", x_train.shape, y_train.shape)
        print("att_mask", att_mask.shape)
        all_sel_tool_ids.extend(sel_tools)

        with tf.GradientTape() as model_tape:
            prediction, att_weights = model([x_train, att_mask], training=True)
            tr_loss, tr_cat_loss = compute_loss(y_train, prediction)
            tr_acc = tf.reduce_mean(compute_acc(y_train, prediction))
        trainable_vars = model.trainable_variables
        model_gradients = model_tape.gradient(tr_loss, trainable_vars)
        enc_optimizer.apply_gradients(zip(model_gradients, trainable_vars))
        epo_tr_batch_loss.append(tr_loss.numpy())
        epo_tr_batch_acc.append(tr_acc.numpy())
        epo_tr_batch_categorical_loss.append(tr_cat_loss.numpy())
        print("Step {}/{}, training binary loss: {}, categorical_loss: {}, training accuracy: {}".format(batch+1, n_train_batches, tr_loss.numpy(), tr_cat_loss.numpy(), tr_acc.numpy()))
        if (batch+1) % test_logging_step == 0:
            print("Predicting on test data...")
            te_loss, te_acc, test_cat_loss, te_prec, low_te_prec = validate_model(test_data, test_labels, model, f_dict, r_dict, u_te_y_labels_dict, trained_on_labels, te_lowest_t_ids)
            epo_te_batch_loss.append(te_loss)
            epo_te_batch_acc.append(te_acc)
            epo_te_batch_categorical_loss.append(test_cat_loss)
            epo_te_precision.append(te_prec)
            epo_low_te_precision.append(low_te_prec)
        print()
        if (batch+1) % train_logging_step == 0:
            print("Saving model at training step {}/{}".format(batch + 1, n_train_batches))
            base_path = "log/saved_model/"
            tf_path = base_path + "{}/".format(batch+1)
            tf_model_save = base_path + "{}/tf_model/".format(batch+1)
            if not os.path.isdir(tf_path):
                os.mkdir(tf_path)
            tf.saved_model.save(model, tf_model_save)

            onnx_model_save = base_path + "{}/onnx_model/".format(batch+1)
            if not os.path.isdir(tf_path):
                os.mkdir(onnx_model_save)
            
            #utils.convert_to_onnx(tf_model_save, onnx_model_save)

    new_dict = dict()
    for k in u_tr_y_labels_dict:
        new_dict[str(k)] = ",".join([str(item) for item in u_tr_y_labels_dict[k]])

    utils.write_file("log/data/epo_tr_batch_loss.txt", ",".join([str(item) for item in epo_tr_batch_loss]))
    utils.write_file("log/data/epo_tr_batch_acc.txt", ",".join([str(item) for item in epo_tr_batch_acc]))
    utils.write_file("log/data/epo_te_batch_loss.txt", ",".join([str(item) for item in epo_te_batch_loss]))
    utils.write_file("log/data/epo_te_batch_acc.txt", ",".join([str(item) for item in epo_te_batch_acc]))
    utils.write_file("log/data/epo_tr_batch_categorical_loss.txt", ",".join([str(item) for item in epo_tr_batch_categorical_loss]))
    utils.write_file("log/data/epo_te_batch_categorical_loss.txt", ",".join([str(item) for item in epo_te_batch_categorical_loss]))
    utils.write_file("log/data/epo_te_precision.txt", ",".join([str(item) for item in epo_te_precision]))
    utils.write_file("log/data/all_sel_tool_ids.txt", ",".join([str(item) for item in all_sel_tool_ids]))
    utils.write_file("log/data/epo_low_te_precision.txt", ",".join([str(item) for item in epo_low_te_precision]))
    utils.write_file("log/data/u_tr_y_labels_dict.txt", new_dict)  
