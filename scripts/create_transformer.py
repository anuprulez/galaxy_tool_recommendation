"""
Create transformers
"""
import os
import time
import random
import numpy as np
import subprocess
import sys

import keras
from keras import backend
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


from scripts import utils
import predict_sequences

'''
Server config

BATCH_SIZE = 128
num_layers = 2
d_model = 64
dff = 256 #2048
num_heads = 4
dropout_rate = 0.5

max_seq_len = 25
index_start_token = 2
logging_step = 5
n_topk = 5

=======

BATCH_SIZE = 128
num_layers = 4
d_model = 64
dff = 256
num_heads = 4
dropout_rate = 0.1

max_seq_len = 25
index_start_token = 2
n_topk = 1

train_logging_step = 5000
test_logging = 100

n_train_batches = 50000
n_test_batches = 10



'''

# TODO: Try out only encoder and sequence classification: https://arxiv.org/pdf/2104.05448.pdf

BATCH_SIZE = 128
num_layers = 4
d_model = 64
dff = 256
num_heads = 4
dropout_rate = 0.5

max_seq_len = 25
index_start_token = 2
n_topk = 1

train_logging_step = 1000
test_logging = 10

n_train_batches = 100000
n_test_batches = 1



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
topk_sce = tf.keras.metrics.SparseTopKCategoricalAccuracy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print('Attention weights are:')
  print(temp_attn)
  print('Output is:')
  print(temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
  def __init__(self,*, num_layers, d_model, num_heads, dff, input_vocab_size,
               rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
        for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
  def __init__(self,*, num_layers, d_model, num_heads, dff, target_vocab_size,
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(target_vocab_size, d_model)

    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


class Transformer(tf.keras.Model):
  def __init__(self,*, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           input_vocab_size=input_vocab_size, rate=rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           target_vocab_size=target_vocab_size, rate=rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
    inp, tar = inputs

    padding_mask, look_ahead_mask = self.create_masks(inp, tar)

    enc_output = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

  def create_masks(self, inp, tar):
    # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
    padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return padding_mask, look_ahead_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss_object, training=False):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  #if training is False:
      #print("Loss function")
      #print(real.shape, pred.shape, tf.argmax(pred, axis=-1).shape)
      #print("Real output: ", real)
      #print("Predicted output: ", tf.argmax(pred, axis=-1))
      #print("Loss: ", tf.reduce_mean(loss_object(real, pred)))
      #print("Accuracy: ", accuracy_function(real, pred))
      #print("-----------")

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def sparse_top_k_categorical_acc(y_true, y_pred, k=5):
  """Creates float Tensor, 1.0 for label-TopK_prediction match, 0.0 for mismatch.
  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    k: (Optional) Number of top elements to look at for computing accuracy.
      Defaults to 5.
  Returns:
    Match tensor: 1.0 for label-prediction match, 0.0 for mismatch.
  """
  reshape_matches = False
  y_true = tf.convert_to_tensor(y_true)
  y_pred = tf.convert_to_tensor(y_pred)
  y_true_rank = y_true.shape.ndims
  y_pred_rank = y_pred.shape.ndims
  y_true_org_shape = tf.shape(y_true)

  # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
  if (y_true_rank is not None) and (y_pred_rank is not None):
    if y_pred_rank > 2:
      y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    if y_true_rank > 1:
      reshape_matches = True
      y_true = tf.reshape(y_true, [-1])

  matches = tf.cast(
      tf.math.in_top_k(
          predictions=y_pred, targets=tf.cast(y_true, 'int32'), k=k),
      dtype=backend.floatx())

  # returned matches is expected to have same shape as y_true input
  if reshape_matches:
    return tf.reshape(matches, shape=y_true_org_shape)

  return matches


def accuracy_function(real, pred):
  topk_pred = tf.math.top_k(pred, k=5, sorted=True)
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  masked_acc = 0
  try:
      #print("In accuracy function ...")
      #print("True First row: ", real[0])
      #print()
      #print("Predicted topk: ", topk_pred.indices[0, 0])
      #topk_acc = topk_sce(real, pred) #tf.math.in_top_k(real, pred, 5, name=None)
      topk_matches = sparse_top_k_categorical_acc(real, pred, n_topk)
      topk_matches = tf.math.logical_not(tf.math.equal(topk_matches, 0))
      #print("Topk matches: ", topk_matches)
      topk_matches = tf.math.logical_and(mask, topk_matches)
      topk_matches = tf.cast(topk_matches, dtype=tf.float32)
      topk_masked_acc = tf.reduce_sum(topk_matches)/tf.reduce_sum(tf.cast(mask, dtype=tf.float32))
      #print("Topk masked batch accuracy: ", topk_masked_acc)
      #print("---------")
  except Exception as e:
      print(e)
      pass
  #accuracies = tf.equal(real, tf.argmax(pred, axis=-1))
  #accuracies = tf.math.logical_and(mask, accuracies)
  #accuracies = tf.cast(accuracies, dtype=tf.float32)
  #mask = tf.cast(mask, dtype=tf.float32)
  #return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
  return topk_masked_acc


def selected_batches(labels):
    freq_dict = dict()
    for item in labels:
        if item not in freq_dict:
            freq_dict[item] = 0
        freq_dict[item] += 1
    freq_dict = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}
    print(freq_dict)
    print()


def sample_test_x_y(X, y):
    rand_batch_indices = np.random.randint(0, X.shape[0], BATCH_SIZE)
    x_batch_train = X[rand_batch_indices]
    y_batch_train = y[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y


def preprocess_sampling_datasets(y_train):
    all_train_labels_seqs = list()
    all_train_labels = list()
    for label in y_train:
        tpos = np.where(label > 0)[0]
        tseq_str = ",".join([str(int(item)) for item in label[1:tpos[-1] + 1]])
        tseq = [int(item) for item in label[1:tpos[-1] + 1]]
        all_train_labels.extend(tseq)
        all_train_labels_seqs.append(tseq_str)
    u_labels = list(set(all_train_labels))
    return u_labels, all_train_labels_seqs


def sample_train_x_y(u_labels, all_train_labels_seqs, X_train, y_train):
    random.shuffle(u_labels)
    b_labels = u_labels[:BATCH_SIZE]
    rand_batch_indices = list()
    for index, seqs in enumerate(all_train_labels_seqs):
        s_split = seqs.split(",")
        s_split = [int(item) for item in s_split]
        intersection = list(set(s_split).intersection(set(b_labels)))
        re_index = [index] * len(intersection)
        b_labels = [e for e in b_labels if e not in intersection]
        if len(re_index) > 0:
            rand_batch_indices.extend(re_index)
        if len(rand_batch_indices) >= BATCH_SIZE:
            break
    rand_batch_indices = rand_batch_indices[:BATCH_SIZE]
    random.shuffle(rand_batch_indices)
    x_batch_train = X_train[rand_batch_indices]
    y_batch_train = y_train[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y


def get_u_labels(y_train):
    s_y_train = y_train[:, 1:2]
    s_y_train = list(s_y_train.reshape(len(s_y_train), ))
    s_y_train = [str(int(item)) for item in s_y_train]
    u_labels = list(set(s_y_train))
    random.shuffle(u_labels)
    return u_labels


def sample_train_x_y_first_pos(u_labels, X_train, y_train, f_dict, rev_dict):
    rand_batch_indices = list()
    b_u_labels = u_labels[:BATCH_SIZE]
    sel_tools = list()
    for index, tar in enumerate(y_train):
        label_val = str(int(tar[1:2]))
        if label_val in b_u_labels:
            sel_tools.append(label_val)
            rand_batch_indices.append(index)
            b_u_labels = [e for e in b_u_labels if e not in [label_val]]
        if len(rand_batch_indices) == BATCH_SIZE:
            break
    x_batch_train = X_train[rand_batch_indices]
    y_batch_train = y_train[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y


def validate_model(b_num, te_input_seqs, te_tar_seqs, te_u_labels, trained_model, f_dict, rev_dict):
    eval_loss = list()
    eval_acc = list()
    print("Test size:", te_input_seqs.shape, te_tar_seqs.shape)
    for i in range(n_test_batches):
        #te_inp, te_tar = sample_train_x_y(te_ulabels, te_all_labels_seq, te_input_seqs, te_tar_seqs)
        #te_inp, te_tar = sample_train_x_y_first_pos(te_u_labels, te_input_seqs, te_tar_seqs, f_dict, rev_dict)
        te_inp, te_tar = sample_test_x_y(te_input_seqs, te_tar_seqs)

        te_tar_inp = te_tar[:, :-1]
        te_tar_real = te_tar[:, 1:]

        te_predictions, _ = trained_model([te_inp, te_tar_inp], training=False)
        te_loss = loss_function(te_tar_real, te_predictions, loss_object, False)
        test_loss(te_loss)
        test_accuracy(accuracy_function(te_tar_real, te_predictions))

        eval_loss.append(test_loss.result())
        eval_acc.append(test_accuracy.result())
    print(f"Test: Batch {b_num+1}/{n_train_batches}: Test Loss {np.mean(eval_loss):.4f}, Test Accuracy {np.mean(eval_acc):.4f}")
    return np.mean(eval_loss), np.mean(eval_acc)


def create_train_model(inp_seqs, tar_seqs, te_input_seqs, te_tar_seqs, f_dict, rev_dict):
    learning_rate = CustomSchedule(d_model, 100)
    #optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    optimizer = tf.keras.optimizers.Adam()

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=len(rev_dict) + 1,
        target_vocab_size=len(rev_dict) + 1,
        rate=dropout_rate)

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
      tar_inp = tar[:, :-1]
      tar_real = tar[:, 1:]

      with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training = True)
        loss = loss_function(tar_real, predictions, loss_object, True)

      gradients = tape.gradient(loss, transformer.trainable_variables)
      optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

      train_loss(loss)
      train_accuracy(accuracy_function(tar_real, predictions))

    #print("Creating datasets for balanced sampling...")
    #tr_ulabels, tr_all_labels_seq = preprocess_sampling_datasets(tar_seqs)
    #te_ulabels, te_all_labels_seq = preprocess_sampling_datasets(te_tar_seqs)

    epo_tr_batch_loss = list()
    epo_tr_batch_acc = list()
    epo_te_batch_loss = list()
    epo_te_batch_acc = list()
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    tr_u_labels = get_u_labels(tar_seqs)
    te_u_labels = get_u_labels(te_tar_seqs)
    print("Unique first labels, tr, te: ", len(tr_u_labels), len(te_u_labels))

    for batch in range(n_train_batches):
        print("Train data size:", inp_seqs.shape, tar_seqs.shape)
        # train on randomly selected samples
        #inp, tar = sample_train_x_y(tr_ulabels, tr_all_labels_seq, inp_seqs, tar_seqs)
        #inp, tar = sample_train_x_y_first_pos(tr_u_labels, inp_seqs, tar_seqs, f_dict, rev_dict)
        inp, tar = sample_test_x_y(inp_seqs, tar_seqs)
        train_step(inp, tar)
        epo_tr_batch_loss.append(train_loss.result().numpy())
        epo_tr_batch_acc.append(train_accuracy.result().numpy())
        print(f'Train: batch {batch+1}/{n_train_batches}: train loss {train_loss.result():.4f}, train accuracy {train_accuracy.result():.4f}')
        if (batch + 1) % test_logging == 0:
            print("-----")
            print("Evaluating on test data...")
            te_loss, te_acc = validate_model(batch, te_input_seqs, te_tar_seqs, te_u_labels, transformer, f_dict, rev_dict)
            epo_te_batch_loss.append(te_loss)
            epo_te_batch_acc.append(te_acc)
            print("-----")
        if (batch + 1) % train_logging_step == 0:
            '''print("Saving model at training step {}/{}".format(batch + 1, n_train_batches))
            base_path = "log/saved_model/"
            tf_path = base_path + "{}/".format(batch+1)
            tf_model_save = base_path + "{}/tf_model/".format(batch+1)
            if not os.path.isdir(tf_path):
                os.mkdir(tf_path)
            tf.saved_model.save(transformer, tf_model_save)
            tf_loaded_model = tf.saved_model.load(tf_model_save)'''
            predictor = predict_sequences.PredictSequence(transformer)
            predictor(f_dict, rev_dict)

    utils.write_file("log/data/epo_tr_batch_loss.txt", ",".join([str(item) for item in epo_tr_batch_loss]))
    utils.write_file("log/data/epo_tr_batch_acc.txt", ",".join([str(item) for item in epo_tr_batch_acc]))
    utils.write_file("log/data/epo_te_batch_loss.txt", ",".join([str(item) for item in epo_te_batch_loss]))
    utils.write_file("log/data/epo_te_batch_acc.txt", ",".join([str(item) for item in epo_te_batch_acc]))

#python_shell_script = "python -m tf2onnx.convert --saved-model " + tf_model_save + " --output " + onnx_model_save + "model.onnx" + " --opset 15 "
#print(python_shell_script)
# convert tf/keras model to ONNX and save it to output file
#subprocess.run(python_shell_script, shell=True, check=True)

#print("Loading ONNX model...")
#loaded_model = onnx.load(onnx_model_save + "model.onnx")
#tf_loaded_model = prepare(loaded_model)
#prediction = tf_loaded_model.run(item, training=False)
#print("Prediction using loaded model...")
'''
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for batch in range(n_train_batches):
            print("Train data size:", inp_seqs.shape, tar_seqs.shape)
            # train on randomly selected samples
            #inp, tar = sample_train_x_y(BATCH_SIZE, tr_ulabels, tr_all_labels_seq, inp_seqs, tar_seqs)
            inp, tar = sample_train_x_y_first_pos(BATCH_SIZE, inp_seqs, tar_seqs)
            train_step(inp, tar)
            epo_tr_batch_loss.append(train_loss.result().numpy())
            epo_tr_batch_acc.append(train_accuracy.result().numpy())
            print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch+1}/{n_train_batches}: Train Loss {train_loss.result():.4f}, Train Accuracy {train_accuracy.result():.4f}')
            if (batch + 1) % test_logging == 0:
                print("Evaluating on test data...")
                #validate_model(epoch, batch, EPOCHS, n_train_batches, BATCH_SIZE, te_input_seqs, te_tar_seqs, te_ulabels, te_all_labels_seq, transformer)
                te_loss, te_acc = validate_model(epoch, batch, EPOCHS, n_train_batches, BATCH_SIZE, te_input_seqs, te_tar_seqs, transformer)
                epo_te_batch_loss.append(te_loss)
                epo_te_batch_acc.append(te_acc)
                print("-----")
        if (epoch + 1) % logging_step == 0:
            print("Saving model at epoch {}/{}".format(epoch+1, EPOCHS))
            base_path = "log/saved_model/"
            tf_path = base_path + "{}/".format(epoch+1)
            tf_model_save = base_path + "{}/tf_model/".format(epoch+1)
            onnx_model_save = base_path + "{}/onnx_model/".format(epoch+1)
            #onnx_path = base_path + "{}/".format(epoch)
            if not os.path.isdir(tf_path):
                os.mkdir(tf_path)
                #os.mkdir(tf_model_save)
                #os.mkdir(onnx_model_save)
            tf.saved_model.save(transformer, tf_model_save)
            tf_loaded_model = tf.saved_model.load(tf_model_save)
            
            predictor = predict_sequences.PredictSequence(transformer)
            predictor(f_dict, rev_dict)
'''
