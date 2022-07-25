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


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
d_dim = 64
dropout = 0.1
n_train_batches = 100
batch_size = 64
test_logging_step = 2

cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


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


def sample_test_x_y(X, y):
    rand_batch_indices = np.random.randint(0, X.shape[0], batch_size)
    x_batch_train = X[rand_batch_indices]
    y_batch_train = y[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y

def loss_func(true, predicted):
    return tf.reduce_mean(bce(true, predicted))


def validate_model(te_x, te_y, model, f_dict, r_dict):
    te_x_batch, y_train_batch = sample_test_x_y(te_x, te_y)
    te_pred_batch, att_weights = model([te_x_batch], training=False)
    for idx in range(te_pred_batch.shape[0]):
        label_pos = np.where(y_train_batch[idx] > 0)[0]
        topk_pred = tf.math.top_k(te_pred_batch[idx], k=len(label_pos), sorted=True)
        topk_pred = topk_pred.indices.numpy()
        print(label_pos, topk_pred)
        label_pos_tools = [r_dict[item] for item in label_pos]
        pred_label_pos_tools = [r_dict[item] for item in topk_pred]
        print(label_pos_tools, pred_label_pos_tools)
        print()
        break

def create_enc_transformer(train_data, train_labels, test_data, test_labels, f_dict, r_dict):

    vocab_size = len(f_dict) + 1
    maxlen = train_data.shape[1]
    enc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

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

    for batch in range(n_train_batches):
        x_train, y_train = sample_test_x_y(train_data, train_labels)
        with tf.GradientTape() as model_tape:
            prediction, att_weights = model([x_train], training=True)
            print(x_train.shape, prediction.shape, att_weights.shape)
            pred_loss = bce(y_train, prediction)
            print(pred_loss)
        print()
        trainable_vars = model.trainable_variables
        model_gradients = model_tape.gradient(pred_loss, trainable_vars)
        enc_optimizer.apply_gradients(zip(model_gradients, trainable_vars))
        print("{} Training loss: {}".format(batch+1, pred_loss))
        if batch+1 % test_logging_step:
            print("Predicting on test data...")
            validate_model(test_data, test_labels, model, f_dict, r_dict)
