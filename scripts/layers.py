import tensorflow as tf
from tensorflow import keras


class ILayer(keras.layers.Layer):
    def __init__(self, max_len):
        super(ILayer, self).__init__()
        self.input_layer = keras.layers.Input(
            shape=(max_len,), dtype='int32'
        )

    def call(self):
        return self.input_layer()


class Embedding(keras.layers.Layer):
    def __init__(self, dimensions, embedding_size, max_len, mask_zero=True):
        super(Embedding, self).__init__()
        self.embedding_layer = keras.layers.Embedding(
            dimensions, embedding_size, input_length=max_len, mask_zero=mask_zero
        )

    def call(self, inputs):
        return self.embedding_layer(inputs)
