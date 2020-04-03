"""
Find the optimal combination of hyperparameters
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

import utils

import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate
from keras.layers import Layer
import keras.backend as K
import kerastuner as kt
from keras import backend as K
import keras.callbacks as callbacks


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class PredictCallback(callbacks.Callback):
    def __init__(self, test_data, test_labels, reverse_data_dictionary, n_epochs, next_compatible_tools, usg_scores):
        self.test_data = test_data
        self.test_labels = test_labels
        self.reverse_data_dictionary = reverse_data_dictionary
        self.precision = list()
        self.usage_weights = list()
        self.n_epochs = n_epochs
        self.next_compatible_tools = next_compatible_tools
        self.pred_usage_scores = usg_scores

    def on_epoch_end(self, epoch, logs={}):
        """
        Compute absolute and compatible precision for test data
        """
        if len(self.test_data) > 0:
            precision, usage_weights = utils.verify_model(self.model, self.test_data, self.test_labels, self.reverse_data_dictionary, self.next_compatible_tools, self.pred_usage_scores)
            self.precision.append(precision)
            self.usage_weights.append(usage_weights)
            print("Epoch %d precision: %s" % (epoch + 1, precision))
            print("Epoch %d usage weights: %s" % (epoch + 1, usage_weights))

class KerasTuneOptimisation:

    @classmethod
    def __init__(self):
        """ Init method. """

    @classmethod
    def train_model(self, config, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights, usage_pred, compatible_next_tools):
        """
        Train a model and report accuracy
        """
        
        l_recurrent_activations = config["activation_recurrent"].split(",")
        l_output_activations = config["activation_output"].split(",")

        # convert items to integer
        l_batch_size = list(map(int, config["batch_size"].split(",")))
        l_embedding_size = list(map(int, config["embedding_size"].split(",")))
        l_units = list(map(int, config["units"].split(",")))

        # convert items to float
        l_learning_rate = list(map(float, config["learning_rate"].split(",")))
        l_dropout = list(map(float, config["dropout"].split(",")))
        l_spatial_dropout = list(map(float, config["spatial_dropout"].split(",")))
        l_recurrent_dropout = list(map(float, config["recurrent_dropout"].split(",")))

        optimize_n_epochs = int(config["optimize_n_epochs"])
        validation_split = float(config["validation_share"])

        # get dimensions
        dimensions = len(reverse_dictionary) + 1
        max_len = 25
        batch_size = 16
        n_epochs = 5
        best_model_params = dict()
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=1e-4)
        predict_callback_test = PredictCallback(test_data, test_labels, reverse_dictionary, n_epochs, compatible_next_tools, usage_pred)

        callbacks_list = [predict_callback_test]
        
        def build_model(hp):
            embedding_size = hp.Int('embedding_size', l_embedding_size[0], l_embedding_size[1])
            gru_units = hp.Int('gru_units', l_units[0], l_units[1])
            sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
            embedded_sequences = tf.keras.layers.Embedding(dimensions, embedding_size, input_length=max_len, mask_zero=True)(sequence_input)
            
            gru = tf.keras.layers.GRU(gru_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )

            sample_output, sample_hidden = gru(embedded_sequences, initial_state=None)

            attention = BahdanauAttention(gru_units)
            context_vector, attention_weights = attention(sample_hidden, sample_output)

            output = tf.keras.layers.Dense(dimensions, activation='sigmoid')(context_vector)

            model = tf.keras.Model(inputs=sequence_input, outputs=output)

            print(model.summary())

            learning_rate = hp.Float('learning_rate', l_learning_rate[0], l_learning_rate[1], sampling='log')
            model.compile(
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate),
                loss=utils.weighted_loss(class_weights),
            )
            return model
      
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=n_epochs,
            hyperband_iterations=optimize_n_epochs,
        )

        tuner.search(
            train_data,
            train_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(test_data, test_labels),
            verbose=1,
            callbacks=callbacks_list
        )

        best_model = tuner.get_best_models(1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        print(best_hyperparameters)

        '''def create_model(params):
            model = Sequential()
            model.add(Embedding(dimensions, int(params["embedding_size"]), mask_zero=True))
            model.add(SpatialDropout1D(params["spatial_dropout"]))
            model.add(GRU(int(params["units"]), dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"], return_sequences=True, activation=params["activation_recurrent"]))
            model.add(Dropout(params["dropout"]))
            model.add(GRU(int(params["units"]), dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"], return_sequences=False, activation=params["activation_recurrent"]))
            model.add(Dropout(params["dropout"]))
            model.add(Dense(dimensions, activation=params["activation_output"]))
            optimizer_rms = RMSprop(lr=params["learning_rate"])
            model.compile(loss=utils.weighted_loss(class_weights), optimizer=optimizer_rms)
            model_fit = model.fit(
                train_data,
                train_labels,
                batch_size=int(params["batch_size"]),
                epochs=optimize_n_epochs,
                shuffle="batch",
                verbose=2,
                validation_split=validation_split,
                callbacks=[early_stopping]
            )
            return {'loss': model_fit.history["val_loss"][-1], 'status': STATUS_OK, 'model': model}
        # minimize the objective function using the set of parameters above
        trials = Trials()
        learned_params = fmin(create_model, params, trials=trials, algo=tpe.suggest, max_evals=int(config["max_evals"]))
        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

        # set the best params with respective values
        for item in learned_params:
            item_val = learned_params[item]
            if item == 'activation_output':
                best_model_params[item] = l_output_activations[item_val]
            elif item == 'activation_recurrent':
                best_model_params[item] = l_recurrent_activations[item_val]
            else:
                best_model_params[item] = item_val
        return best_model_params, best_model'''
