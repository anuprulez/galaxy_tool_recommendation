"""
Find the optimal combination of hyperparameters
"""

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

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

import bahdanau_attention
import utils


class HyperparameterOptimisation:

    @classmethod
    def __init__(self):
        """ Init method. """

    @classmethod
    def train_model(self, config, reverse_dictionary, train_data, train_labels, class_weights):
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
        max_len = train_data.shape[1]
        best_model_params = dict()
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=1e-4)

        # specify the search space for finding the best combination of parameters using Bayesian optimisation
        params = {
            "embedding_size": hp.quniform("embedding_size", l_embedding_size[0], l_embedding_size[1], 1),
            "units": hp.quniform("units", l_units[0], l_units[1], 1),
            "batch_size": hp.quniform("batch_size", l_batch_size[0], l_batch_size[1], 1),
            "activation_recurrent": hp.choice("activation_recurrent", l_recurrent_activations),
            "activation_output": hp.choice("activation_output", l_output_activations),
            "learning_rate": hp.loguniform("learning_rate", np.log(l_learning_rate[0]), np.log(l_learning_rate[1])),
            "dropout": hp.uniform("dropout", l_dropout[0], l_dropout[1]),
            "spatial_dropout": hp.uniform("spatial_dropout", l_spatial_dropout[0], l_spatial_dropout[1]),
            "recurrent_dropout": hp.uniform("recurrent_dropout", l_recurrent_dropout[0], l_recurrent_dropout[1])
        }
        
        def create_model(params):
            embedding_size = int(params["embedding_size"])
            gru_units = int(params["units"])
            sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
            embedded_sequences = tf.keras.layers.Embedding(dimensions, embedding_size, input_length=max_len, mask_zero=True)(sequence_input)

            gru = tf.keras.layers.GRU(gru_units,
                return_sequences=True,
                return_state=True,
                activation='elu'
            )

            sample_output, sample_hidden = gru(embedded_sequences, initial_state=None)

            attention = bahdanau_attention.BahdanauAttention(gru_units)
            context_vector, attention_weights = attention(sample_hidden, sample_output)

            output = tf.keras.layers.Dense(dimensions, activation='sigmoid')(context_vector)

            model = tf.keras.Model(inputs=sequence_input, outputs=output)

            print(model.summary())

            learning_rate = params["learning_rate"]
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                loss=utils.weighted_loss(class_weights),
            )
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
        return best_model_params, best_model
