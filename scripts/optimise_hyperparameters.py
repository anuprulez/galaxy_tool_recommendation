"""
Find the optimal combination of hyperparameters
"""

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

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
        l_dense_activation = config["dense_activation"].split(",")
        l_output_activation = config["output_activation"].split(",")

        # convert items to integer
        l_batch_size = list(map(int, config["batch_size"].split(",")))
        l_embedding_size = list(map(int, config["embedding_size"].split(",")))
        l_deep_size = list(map(int, config["dense_size"].split(",")))
        l_kernel_size = list(map(int, config["kernel_size"].split(",")))
        l_filter_size = list(map(int, config["filter_size"].split(",")))

        # convert items to float
        l_learning_rate = list(map(float, config["learning_rate"].split(",")))
        l_dropout = list(map(float, config["dropout"].split(",")))
        l_spatial_dropout = list(map(float, config["spatial_dropout"].split(",")))

        optimize_n_epochs = int(config["optimize_n_epochs"])
        validation_split = float(config["validation_share"])

        # get dimensions
        dimensions = len(reverse_dictionary) + 1
        best_model_params = dict()
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-4, verbose=1, patience=1)

        # specify the search space for finding the best combination of parameters using Bayesian optimisation
        params = {
            "embedding_size": hp.quniform("embedding_size", l_embedding_size[0], l_embedding_size[1], 1),
            "dense_size": hp.quniform("dense_size", l_deep_size[0], l_deep_size[1], 1),
            "batch_size": hp.quniform("batch_size", l_batch_size[0], l_batch_size[1], 1),
            "kernel_size": hp.quniform("kernel_size", l_kernel_size[0], l_kernel_size[1], 1),
            "filter_size": hp.quniform("filter_size", l_filter_size[0], l_filter_size[1], 1),
            "dense_activation": hp.choice("dense_activation", l_dense_activation),
            "output_activation": hp.choice("output_activation", l_output_activation),
            "learning_rate": hp.loguniform("learning_rate", np.log(l_learning_rate[0]), np.log(l_learning_rate[1])),
            "dropout": hp.uniform("dropout", l_dropout[0], l_dropout[1]),
            "spatial_dropout": hp.uniform("spatial_dropout", l_spatial_dropout[0], l_spatial_dropout[1]),
        }

        def create_model(params):
            model = Sequential()
            model.add(Embedding(dimensions, int(params["embedding_size"])))
            model.add(SpatialDropout1D(params["spatial_dropout"]))
            model.add(Conv1D(int(params["filter_size"]), int(params["kernel_size"]), activation=params['dense_activation']))
            model.add(Dropout(params["dropout"]))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(int(params["dense_size"]), activation=params['dense_activation']))
            model.add(Dense(dimensions, activation=params['output_activation']))
            model.compile(loss=utils.weighted_loss(class_weights), optimizer=RMSprop(lr=params["learning_rate"]))
            model_fit = model.fit(
                train_data,
                train_labels,
                batch_size=int(params["batch_size"]),
                epochs=optimize_n_epochs,
                shuffle="batch",
                verbose=2,
                validation_split=validation_split,
                #callbacks=[early_stopping]
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
                best_model_params[item] = l_output_activation[item_val]
            elif item == 'output_activation':
                best_model_params[item] = l_dense_activation[item_val]
            else:
                best_model_params[item] = item_val
        return best_model_params, best_model
