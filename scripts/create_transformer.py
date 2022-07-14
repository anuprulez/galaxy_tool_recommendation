"""
Find the optimal combination of hyperparameters
"""

import numpy as np
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding, SpatialDropout1D
#from tensorflow.keras.layers.embeddings import 
#from keras.layers.core import 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from scripts import utils


BATCH_SIZE = 64
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 200
max_seq_len = 25
index_start_token = 2


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


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


def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=-1))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def sample_true_x_y(batch_size, X_train, y_train):
    rand_batch_indices = np.random.randint(0, X_train.shape[0], batch_size)
    x_batch_train = X_train[rand_batch_indices]
    y_batch_train = y_train[rand_batch_indices]
    unrolled_x = x_batch_train #utils.convert_to_array(x_batch_train)
    unrolled_y = y_batch_train #utils.convert_to_array(y_batch_train)
    return unrolled_x, unrolled_y


def create_sample_test_data(f_dict):
    tool_name = "bam_to_sam" #"Summary_Statistics1"
    #seq2 = "bowtie2"
    input_mat = np.zeros([1, 25])
    tool_id = f_dict[tool_name]
    print(tool_name, tool_id)
    input_mat[:, max_seq_len - 1:max_seq_len] = tool_id
    return input_mat


def create_train_model(inp_seqs, tar_seqs, te_input_seqs, te_tar_seqs, f_dict, rev_dict):
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
   
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=len(rev_dict) + 1,
        target_vocab_size=len(rev_dict) + 1,
        rate=dropout_rate)

    n_train_batches = int(inp_seqs.shape[0] / float(BATCH_SIZE))

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for batch in range(n_train_batches):
            inp, tar = sample_true_x_y(BATCH_SIZE, inp_seqs, tar_seqs)
            te_inp, te_tar = sample_true_x_y(BATCH_SIZE, te_input_seqs, te_tar_seqs)

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            print("inp:", inp[0])
            print("Target: ", tar[0])
            print("tar_inp: ", tar_inp[0])
            print("tar_real: ", tar_real[0])

            te_tar_inp = te_tar[:, :-1]
            te_tar_real = te_tar[:, 1:]

            with tf.GradientTape() as tape:
                predictions, _ = transformer([inp, tar_inp], training = True)
                te_predictions, _ = transformer([te_inp, te_tar_inp], training = False)
                pred_argmax = tf.argmax(te_predictions, axis=-1)
                pred_seq = pred_argmax[0,:]
                loss = loss_function(tar_real, predictions, loss_object)
                te_loss = loss_function(te_tar_real, te_predictions, loss_object)
            print("Pred seq: ", pred_seq)
            print()
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            test_loss(te_loss)
            train_accuracy(accuracy_function(tar_real, predictions))
            test_accuracy(accuracy_function(te_tar_real, te_predictions))

            #if batch % 50 == 0:
            #print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            print()
            print(f'Epoch {epoch + 1}, Batch {batch + 1}: Train Loss {train_loss.result():.4f}, Train Accuracy {train_accuracy.result():.4f}')
            print(f'Epoch {epoch + 1}, Batch {batch + 1}: Test Loss {test_loss.result():.4f}, Test Accuracy {test_accuracy.result():.4f}')
            #print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


            if batch % 50 == 0:
                translator = Translator(transformer)
                #test_input_seq = create_sample_test_data(f_dict)
                #print(tf.constant(test_input_seq))
                #translated_text, translated_tokens, attention_weights = translator(tf.constant(test_input_seq), f_dict, rev_dict)
                #translator(te_inp, te_tar, f_dict, rev_dict)
                translator(te_inp, te_tar, f_dict, rev_dict)


class Translator(tf.Module):
  def __init__(self, transformer):
    #self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, te_inp, te_tar, f_dict, rev_dict):
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

    #sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
    encoder_input = sentence
    print("True input seq: ", encoder_input)
    # As the output language is english, initialize the output with the
    # english start token.
    #start_end = self.tokenizers.en.tokenize([''])[0]
    #start = start_end[0][tf.newaxis]
    #end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    #output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    
    # working code
    '''output_array = tf.zeros([1, max_seq_len], tf.int64) #output_array.write(0, 0)
    predictions, _ = self.transformer([encoder_input, output_array], training=False)
    #print(predictions.shape)
    predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
    #print(predictions.shape)
    predicted_id = tf.argmax(predictions, axis=-1)
    pred_val = predicted_id.numpy()[0][0]
    print(rev_dict[pred_val], pred_val)'''

    #print("true target seq: ", te_tar[0])
    print()

    # loop over target
    target_seq = te_tar[0]
    n_target_items = np.where(target_seq > 0)[0]
    #output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    #output_array = output_array.write(0, 0)
    '''for i in range(max_seq_len):
        output_array = output_array.write(i, tf.constant(0, dtype=tf.int64))'''
    np_output_array = np.zeros((1, max_seq_len))
    #output_array = tf.reshape(output_array, [1, max_seq_len])
    np_output_array[:, 0] = start_token
    print(np_output_array)
    #print(n_target_items)
    print("Looping predictions ...")
    for i, index in enumerate(n_target_items):
        print(i, index)
        output = tf.constant(np_output_array) #tf.transpose(output_array.stack())
        #output = tf.reshape(output, [1, max_seq_len])
        #print(output)
        orig_predictions, _ = self.transformer([encoder_input, output], training=False)
        print("true target seq: ", te_tar[0])
        print("Pred seq argmax: ", tf.argmax(orig_predictions, axis=-1))
        predictions = orig_predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        #print("Last predicted id:", predicted_id[0][0])
        np_output_array[:, index+1] = predicted_id[0][0]
        print(np_output_array)
        print("----------")

    print("Overall loss and accuracy:")
    #target_seq = tf.reshape(te_tar[0], [1, max_seq_len])
    #print(target_seq, orig_predictions)
    prediction_loss = loss_function(target_seq, orig_predictions, loss_object)
    test_loss(prediction_loss)
    test_accuracy(accuracy_function(target_seq, orig_predictions))
    print(f'Prediction Test: Loss {test_loss.result():.4f}, Accuracy {test_accuracy.result():.4f}')
    print()
    '''for i in tf.range(max_seq_len):
      #output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output_array], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)
      pred_val = predicted_id.numpy()[0][0]
      print(i, pred_val, rev_dict[pred_val])'''
      #print()
      
      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      #output_array = output_array.write(i+1, predicted_id[0])

      #print(output_array)

      #if predicted_id == end:
      #  break

    #output = tf.transpose(output_array.stack())

    #print(output_array)
    # output.shape (1, tokens)
    #text = tokenizers.en.detokenize(output)[0]  # shape: ()

    #tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    #_, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    #return text, tokens, attention_weights



'''train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))'''
