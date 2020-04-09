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
        self.spatial_dropout = parameters["spatial_dropout"]
        self.recurrent_dropout = parameters["recurrent_dropout"]
        self.dropout = parameters["dropout"]
        self.clip_norm = parameters["clip_norm"]

    def create_model(self):
        sequence_input = tf.keras.layers.Input(shape=(self.max_len,), dtype='int32')
        embedded_sequences = tf.keras.layers.Embedding(self.dimensions, self.embedding_size, input_length=self.max_len, mask_zero=True)(sequence_input)
        embedded_sequences = tf.keras.layers.SpatialDropout1D(self.spatial_dropout)(embedded_sequences)
        gru_1 = tf.keras.layers.GRU(self.gru_units,
            return_sequences=True,
            return_state=False,
            activation='elu',
            recurrent_dropout=self.recurrent_dropout
        )
        gru_2 = tf.keras.layers.GRU(self.gru_units,
            return_sequences=True,
            return_state=True,
            activation='elu',
            recurrent_dropout=self.recurrent_dropout
        )
        gru_output = gru_1(embedded_sequences)
        gru_output = tf.keras.layers.Dropout(self.dropout)(gru_output)
        gru_output, gru_hidden = gru_2(gru_output)
        attention = bahdanau_attention.BahdanauAttention(self.gru_units)
        context_vector, attention_weights = attention(gru_hidden, gru_output)
        dropout = tf.keras.layers.Dropout(self.dropout)(context_vector)
        output = tf.keras.layers.Dense(self.dimensions, activation='sigmoid')(dropout)
        model = tf.keras.Model(inputs=sequence_input, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, clipnorm=self.clip_norm, centered=True),
            loss=utils.weighted_loss(self.class_weights),
        )
        return model
