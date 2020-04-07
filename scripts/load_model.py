import tensorflow as tf

import bahdanau_attention
import utils


class ToolPredictionAttentionModel():
  
    def __init__(self, parameters):
        self.embedding_size = int(parameters["embedding_size"])
        self.gru_units = int(parameters["units"])
        self.max_len = parameters["max_len"]
        self.dimensions = parameters["dimensions"]
        self.learning_rate = parameters["learning_rate"]
        self.class_weights = parameters["class_weights"]

    def create_model(self):
        sequence_input = tf.keras.layers.Input(shape=(self.max_len,), dtype='int32')
        embedded_sequences = tf.keras.layers.Embedding(self.dimensions, self.embedding_size, input_length=self.max_len, mask_zero=True)(sequence_input)
        gru = tf.keras.layers.GRU(self.gru_units,
            return_sequences=True,
            return_state=True,
            activation='elu'
        )
        sample_output, sample_hidden = gru(embedded_sequences, initial_state=None)
        attention = bahdanau_attention.BahdanauAttention(self.gru_units)
        context_vector, attention_weights = attention(sample_hidden, sample_output)
        output = tf.keras.layers.Dense(self.dimensions, activation='sigmoid')(context_vector)
        model = tf.keras.Model(inputs=sequence_input, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
            loss=utils.weighted_loss(self.class_weights),
        )
        return model
