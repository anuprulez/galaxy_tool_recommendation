"""
Find the optimal combination of hyperparameters
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import kerastuner as kt

import bahdanau_attention
import utils


class KerasTuneOptimisation:

    @classmethod
    def __init__(self, n_cpus):
        """ Init method. """
        tf.config.threading.set_inter_op_parallelism_threads(n_cpus)
        tf.config.threading.set_intra_op_parallelism_threads(n_cpus)

    @classmethod
    def optimise_parameters(self, config, data_dictionary, reverse_dictionary, train_data, train_labels, class_weights, compatible_next_tools):
        """
        Train a model and report accuracy
        """
        
        # convert items to integer
        l_batch_size = list(map(int, config["batch_size"].split(",")))
        l_embedding_size = list(map(int, config["embedding_size"].split(",")))
        l_units = list(map(int, config["units"].split(",")))

        # convert items to float
        l_learning_rate = list(map(float, config["learning_rate"].split(",")))
        l_dropout = list(map(float, config["dropout"].split(",")))
        l_spatial_dropout = list(map(float, config["spatial_dropout"].split(",")))
        l_recurrent_dropout = list(map(float, config["recurrent_dropout"].split(",")))

        n_epochs = int(config["n_epochs"])
        optimize_n_epochs = int(config["optimize_n_epochs"])
        max_evals = int(config["max_evals"])
        validation_split = float(config["validation_share"])

        # get dimensions
        dimensions = len(reverse_dictionary) + 1
        max_len = train_data.shape[1]
        batch_size = l_batch_size[0]
        best_model_params = dict()
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=1e-1)

        callbacks_list = [early_stopping]
        
        def build_model(hp):
            embedding_size = hp.Int('embedding_size', l_embedding_size[0], l_embedding_size[1], step=10)
            gru_units = hp.Int('gru_units', l_units[0], l_units[1], step=10)
            spatial_dropout = hp.Float('spatial_dropout', l_spatial_dropout[0], l_spatial_dropout[1], step=0.05)
            dropout = hp.Float('dropout', l_dropout[0], l_dropout[1], step=0.05)
            recurrent_dropout = hp.Float('recurrent_dropout', l_recurrent_dropout[0], l_recurrent_dropout[1], step=0.05)
            
            sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
            
            embedded_sequences = tf.keras.layers.Embedding(dimensions, embedding_size, input_length=max_len, mask_zero=True)(sequence_input)
            
            embedded_sequences = tf.keras.layers.SpatialDropout1D(spatial_dropout)(embedded_sequences)
            
            gru_1 = tf.keras.layers.GRU(gru_units,
                return_sequences=True,
                return_state=False,
                activation='elu',
                recurrent_dropout=recurrent_dropout
            )
            
            gru_2 = tf.keras.layers.GRU(gru_units,
                return_sequences=True,
                return_state=True,
                activation='elu',
                recurrent_dropout=recurrent_dropout
            )

            gru_output = gru_1(embedded_sequences)
            
            gru_output = tf.keras.layers.Dropout(dropout)(gru_output)
            
            gru_output, gru_hidden = gru_2(gru_output)

            attention = bahdanau_attention.BahdanauAttention(gru_units)
            context_vector, attention_weights = attention(gru_hidden, gru_output)

            context_vector = tf.keras.layers.Dropout(dropout)(context_vector)

            output = tf.keras.layers.Dense(dimensions, activation='sigmoid')(context_vector)

            model = tf.keras.Model(inputs=sequence_input, outputs=output)

            print(model.summary())

            learning_rate = hp.Float('learning_rate', l_learning_rate[0], l_learning_rate[1], sampling='log')
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                loss=utils.weighted_loss(class_weights),
            )
            return model
      
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            factor=2,
            max_epochs=max_evals,
            hyperband_iterations=optimize_n_epochs,
            directory='opt_trials',
            project_name='tool_prediction'
        )

        tuner.search(
            train_data,
            train_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=callbacks_list
        )

        tuner.results_summary()
        best_model = tuner.get_best_models(1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        best_hyperparameters = best_hyperparameters.get_config()["values"]
        best_hyperparameters["batch_size"] = batch_size
        opt_results = {
            "model": best_model,
            "best_parameters": best_hyperparameters
        }
        utils.save_model(opt_results, data_dictionary, compatible_next_tools, config["trained_model_path"], class_weights, max_len)
        return best_hyperparameters, best_model
