import os
import time
import random
import numpy as np
import subprocess
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model


embed_dim = 128 # Embedding size for each token
num_heads = 4 # Number of attention heads
ff_dim = 64 # Hidden layer size in feed forward network inside transformer
d_dim = 64
dropout = 0.2
n_train_batches = 10000
batch_size = 32
test_logging_step = 50
train_logging_step = 200
learning_rate = 1e-2

cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bca = tf.keras.metrics.CategoricalAccuracy()


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output, attention_scores = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attention_scores



class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def get_u_labels(y_train):
    last_tools = list()
    for item in range(y_train.shape[0]):
        arr_seq = y_train[item]
        #print(arr_seq)
        label_pos = np.where(arr_seq > 0)[0]
        #print(label_pos, arr_seq)
        last_tool = str(int(arr_seq[label_pos[-1]]))
        seq = ",".join([str(int(a)) for a in arr_seq[0:label_pos[-1] + 1]])
        #print(seq, last_tool)
        last_tools.append(last_tool)
        #print()
    u_labels = list(set(last_tools))
    #print(len(last_tools), len(u_labels))
    random.shuffle(u_labels)
    return u_labels


def sample_test_x_y(X, y):
    rand_batch_indices = np.random.randint(0, X.shape[0], batch_size)
    x_batch_train = X[rand_batch_indices]
    y_batch_train = y[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y


def loss_func(true, predicted):
    return tf.reduce_mean(bce(true, predicted))


def sample_balanced(x_seqs, y_labels, ulabels):
    bat_ulabels = ulabels[:batch_size]
    rand_batch_indices = list()
    #print(bat_ulabels)
    for item in range(x_seqs.shape[0]):
        arr_seq = x_seqs[item]
        label_pos = np.where(arr_seq > 0)[0]
        last_tool = str(int(arr_seq[label_pos[-1]]))
        if last_tool in bat_ulabels:
            #print(arr_seq, last_tool)
            rand_batch_indices.append(item)
            bat_ulabels = [e for e in bat_ulabels if e not in [last_tool]]
            #print(bat_ulabels, len(bat_ulabels))
            #print()
        if len(rand_batch_indices) == batch_size:
            break
    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y
        

def validate_model(te_x, te_y, model, f_dict, r_dict, u_te_labels):
    #te_x_batch, y_train_batch = sample_test_x_y(te_x, te_y)
    te_x_batch, y_train_batch = sample_balanced(te_x, te_y, u_te_labels)
    te_pred_batch, att_weights = model([te_x_batch], training=False)
    for idx in range(te_pred_batch.shape[0]):
        print(te_x_batch[idx])
        label_pos = np.where(y_train_batch[idx] > 0)[0]
        topk_pred = tf.math.top_k(te_pred_batch[idx], k=len(label_pos), sorted=True)
        topk_pred = topk_pred.indices.numpy()
        print(label_pos, topk_pred)
        label_pos_tools = [r_dict[str(item)] for item in label_pos]
        pred_label_pos_tools = [r_dict[str(item)] for item in topk_pred]
        print(label_pos_tools, pred_label_pos_tools)
        print()
    print("Test finished")

def create_enc_transformer(train_data, train_labels, test_data, test_labels, f_dict, r_dict):

    vocab_size = len(f_dict) + 1
    maxlen = train_data.shape[1]
    enc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x, weights = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(d_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=[outputs, weights])
    u_train_labels = get_u_labels(train_data)
    u_te_labels = get_u_labels(test_data)
    x_train, y_train = sample_balanced(train_data, train_labels, u_train_labels)
    #sys.exit()
    for batch in range(n_train_batches):
        #x_train, y_train = sample_test_x_y(train_data, train_labels)
        x_train, y_train = sample_balanced(train_data, train_labels, u_train_labels)
        #sys.exit()
        with tf.GradientTape() as model_tape:
            prediction, att_weights = model([x_train], training=True)
            pred_loss = bce(y_train, prediction)
            tr_acc = bca(y_train, prediction)
        #print()
        trainable_vars = model.trainable_variables
        model_gradients = model_tape.gradient(pred_loss, trainable_vars)
        enc_optimizer.apply_gradients(zip(model_gradients, trainable_vars))
        print("Step {}/{}, training loss: {}, training accuracy: {}".format(batch+1, n_train_batches, pred_loss.numpy(), tr_acc.numpy()))
        if (batch+1) % test_logging_step == 0:
            print("Predicting on test data...")
            validate_model(test_data, test_labels, model, f_dict, r_dict, u_te_labels)
        print()
        if (batch+1) % train_logging_step == 0:
            print("Saving model at training step {}/{}".format(batch + 1, n_train_batches))
            base_path = "log/saved_model/"
            tf_path = base_path + "{}/".format(batch+1)
            tf_model_save = base_path + "{}/tf_model/".format(batch+1)
            if not os.path.isdir(tf_path):
                os.mkdir(tf_path)
            tf.saved_model.save(model, tf_model_save)
