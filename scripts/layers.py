import tensorflow as tf
from tensorflow import keras


class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, dimensions, embedding_size, max_len, mask_zero=True):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layer = keras.layers.Embedding(
            dimensions, embedding_size, input_length=max_len, mask_zero=mask_zero
        )
        
    def get_config(self):
        return super().get_config().copy()

    def call(self, inputs):
        return self.embedding_layer(inputs)


class SpatialDropoutLayer(keras.layers.Layer):
    def __init__(self, s_dropout):
        super(SpatialDropoutLayer, self).__init__()
        self.spatial_dropout_layer = keras.layers.SpatialDropout1D(
            s_dropout
        )

    def get_config(self):
        return super().get_config().copy()
    
    def call(self, inputs):
        return self.spatial_dropout_layer(inputs)

  
class DropoutLayer(keras.layers.Layer):
    def __init__(self, dropout):
        super(DropoutLayer, self).__init__()
        self.dropout_layer = keras.layers.Dropout(
            dropout
        )
        
    def get_config(self):
        return super().get_config().copy()

    def call(self, inputs):
        return self.dropout_layer(inputs)


class GRULayer(keras.layers.Layer):
    def __init__(self, gru_units, return_sequences=False, return_state=False, activation='tanh', recurrent_dropout='sigmoid'):
        super(GRULayer, self).__init__()
        self.gru_layer = tf.keras.layers.GRU(
            gru_units,
            return_sequences=return_sequences,
            return_state=return_state,
            activation=activation,
            recurrent_dropout=recurrent_dropout
        )
    
    def get_config(self):
        return super().get_config().copy()
    
    def call(self, inputs):
        return self.gru_layer(inputs)


class DenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(DenseLayer, self).__init__()
        self.dense_layer = keras.layers.Dense(
            units,
            activation=activation,
        )
    
    def get_config(self):
        return super().get_config().copy()
    
    def call(self, inputs):
        return self.dense_layer(inputs)
