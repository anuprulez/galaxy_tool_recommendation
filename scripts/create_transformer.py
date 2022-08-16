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
from tensorflow.keras.layers import Dense, GRU, GlobalAveragePooling1D, Dropout, Embedding, SpatialDropout1D
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


=========
https://www.tensorflow.org/text/tutorials/transformer

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1


'''

# TODO: Try out only encoder and sequence classification: https://arxiv.org/pdf/2104.05448.pdf

batch_size = 512
num_layers = 1
d_model = 128
dff = 128
num_heads = 4
#n_train_batches = 40000
dropout_rate = 0.1

max_seq_len = 25
index_start_token = 2
n_topk = 1

train_logging_step = 1000
test_logging_step = 10
te_batch_size = batch_size

n_train_batches = 100000
n_test_batches = 1
learning_rate = 1e-3 #2e-5 #1e-3


#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#topk_sce = tf.keras.metrics.SparseTopKCategoricalAccuracy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


binary_ce = tf.keras.losses.BinaryCrossentropy()
binary_fce = tf.keras.losses.BinaryFocalCrossentropy(gamma=5)
binary_acc = tf.keras.metrics.BinaryAccuracy()
categorical_ce = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)


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
    #print("Before In Split_heads:", x.shape)
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    #print("After In Split_heads:", x.shape)
    #print(tf.transpose(x, perm=[0, 2, 1, 3]).shape)
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    #batch_size = tf.shape(q)[0]

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
    #print("In MHA:", x.shape, mask.shape)
    attn_output, attention_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    #print("Out In MHA:", attn_output.shape, attention_weights.shape)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    #print("Out 2 In MHA:", out2.shape, attention_weights.shape)
    return out2, attention_weights


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    # Embedding(input_dim=vocab_size, output_dim=embed_dim)
    self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
        for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    #print("before x.shape: ", x.shape, mask.shape)
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    #print("before MHA x.shape: ", x.shape, mask.shape)
    for i in range(self.num_layers):
      x, att_weights = self.enc_layers[i](x, training, mask)
    #print("After encoder layer x.shape: ", x.shape)

    x = GlobalAveragePooling1D()(x)
    #print("After GlobalAveragePooling1D x.shape: ", x.shape)
    return x, att_weights  # (batch_size, input_seq_len, d_model)


class Transformer(tf.keras.Model):
  def __init__(self,*, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
    super().__init__()

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           input_vocab_size=input_vocab_size, rate=rate)

    self.final_layer = tf.keras.layers.Dense(input_vocab_size)


  def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument

    padding_mask = tf.cast(tf.math.logical_not(tf.math.equal(inputs, 0)), dtype=tf.float32) #self.create_masks(inputs)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
    #padding_mask = self.create_masks(inputs)
    '''print(inputs)
    print()
    print(padding_mask)
    print()
    print(tf.math.logical_not(tf.math.equal(inputs, 0)))'''
    enc_output, enc_attention = self.encoder(inputs, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    '''dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, padding_mask)'''

    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, enc_attention

  def create_masks(self, inp):
    # Encoder padding mask (Used in the 2nd attention block in the decoder too.)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    '''look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)'''

    return create_padding_mask(inp)


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


def get_u_labels(y_train):
    last_tools = list()
    ulabels_dict = dict()
    for item in range(y_train.shape[0]):
        arr_seq = y_train[item]
        label_pos = np.where(arr_seq > 0)[0]
        last_tool = str(int(arr_seq[label_pos[-1]]))
        if last_tool not in ulabels_dict:
            ulabels_dict[last_tool] = list()
        ulabels_dict[last_tool].append(item)
        seq = ",".join([str(int(a)) for a in arr_seq[0:label_pos[-1] + 1]])
        last_tools.append(last_tool)
    u_labels = list(set(last_tools))
    random.shuffle(u_labels)
    return u_labels, ulabels_dict


def sample_test_x_y(X, y):
    rand_batch_indices = np.random.randint(0, X.shape[0], batch_size)
    x_batch_train = X[rand_batch_indices]
    y_batch_train = y[rand_batch_indices]
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


def create_model(num_layers, d_model, num_head, dff, rev_dict, dropout_rate=0.1):

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=len(rev_dict) + 1,
        rate=dropout_rate)

    return transformer


def validate_model(te_x, te_y, model, f_dict, r_dict, ulabels_te_dict, tr_labels, lowest_t_ids):
    #te_x_batch, y_train_batch = sample_test_x_y(te_x, te_y)
    #te_x_batch, y_train_batch = sample_balanced(te_x, te_y, ulabels_te_dict)
    print("Total test data size: ", te_x.shape, te_y.shape)
    te_x_batch, y_train_batch, _ = sample_balanced_te_y(te_x, te_y, ulabels_te_dict, te_batch_size)
    print("Batch test data size: ", te_x_batch.shape, y_train_batch.shape)
    te_pred_batch, att_weights = model(te_x_batch, training=False)
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

    print("# Test lowest ids", len(lowest_t_ids))
    low_te_data = te_x[lowest_t_ids]
    low_te_labels = te_y[lowest_t_ids]
    
    low_te_data = low_te_data[:batch_size]
    low_te_labels = low_te_labels[:batch_size]

    low_te_pred_batch, low_att_weights = model(low_te_data, training=False)
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


def create_train_model(inp_seqs, tar_seqs, te_input_seqs, te_tar_seqs, f_dict, rev_dict, c_wts, tr_t_freq):
    #learning_rate = CustomSchedule(d_model, 1000)
    #enc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    enc_optimizer = tf.keras.optimizers.Adam()
    print(inp_seqs.shape, tar_seqs.shape, te_input_seqs.shape, te_tar_seqs.shape)
    all_sel_tool_ids = list()

    transformer = create_model(num_layers, d_model, num_heads, dff, rev_dict)

    u_tr_y_labels, u_tr_y_labels_dict = get_u_tr_labels(tar_seqs)
    u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(te_tar_seqs)

    trained_on_labels = [int(item) for item in list(u_tr_y_labels_dict.keys())]

    te_lowest_t_ids = utils.get_low_freq_te_samples(te_input_seqs, te_tar_seqs, tr_t_freq)
    utils.write_file("log/data/te_lowest_t_ids.txt", ",".join([str(item) for item in te_lowest_t_ids]))

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

    tr_u_labels = get_u_labels(tar_seqs)
    te_u_labels = get_u_labels(te_tar_seqs)

    print("Unique first labels, tr, te: ", len(tr_u_labels), len(te_u_labels))
    sel_tools = list()

    for batch in range(n_train_batches):
        print("Train data size:", inp_seqs.shape, tar_seqs.shape)
        #x_train, y_train, sel_tools = sample_balanced_tr_y(inp_seqs, tar_seqs, u_tr_y_labels_dict, batch_size, tr_t_freq, sel_tools)
        x_train, y_train, sel_tools = sample_balanced_te_y(inp_seqs, tar_seqs, u_tr_y_labels_dict, batch_size)
        print("Batch train data size: ", x_train.shape, y_train.shape)
        all_sel_tool_ids.extend(sel_tools)
        #inp, tar = sample_test_x_y(te_input_seqs, te_tar_seqs)
        with tf.GradientTape() as transformer_tape:
            #inputs = [x_train, y_train]
            prediction, att_weights = transformer(x_train, training=True)
            #print(y_train.shape, prediction.shape, att_weights.shape)
            tr_loss, tr_cat_loss = compute_loss(y_train, prediction)
            tr_acc = tf.reduce_mean(compute_acc(y_train, prediction))

        trainable_vars = transformer.trainable_variables
        transformer_gradients = transformer_tape.gradient(tr_loss, trainable_vars)
        enc_optimizer.apply_gradients(zip(transformer_gradients, trainable_vars))

        epo_tr_batch_loss.append(tr_loss.numpy())
        epo_tr_batch_acc.append(tr_acc.numpy())
        epo_tr_batch_categorical_loss.append(tr_cat_loss.numpy())
        print("Step {}/{}, training binary loss: {}, categorical_loss: {}, training accuracy: {}".format(batch+1, n_train_batches, tr_loss.numpy(), tr_cat_loss.numpy(), tr_acc.numpy()))
        if (batch+1) % test_logging_step == 0:
            print("Predicting on test data...")
            te_loss, te_acc, test_cat_loss, te_prec, low_te_prec = validate_model(te_input_seqs, te_tar_seqs, transformer, f_dict, rev_dict, u_te_y_labels_dict, trained_on_labels, te_lowest_t_ids)
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
            tf.saved_model.save(transformer, tf_model_save)

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

'''
