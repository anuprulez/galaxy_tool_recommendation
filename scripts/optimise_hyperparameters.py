"""
Find the optimal combination of hyperparameters
"""

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import utils


class HyperparameterOptimisation:

    def __init__(self):
        """ Init method. """

    def train_model(self, config, reverse_dictionary, train_data, train_labels, test_data, test_labels, tool_tr_samples, class_weights):
        """
        Train a model and report accuracy
        """
        # convert items to integer
        '''l_batch_size = list(map(int, config["batch_size"].split(",")))
        l_embedding_size = list(map(int, config["embedding_size"].split(",")))
        l_units = list(map(int, config["units"].split(",")))

        # convert items to float
        l_learning_rate = list(map(float, config["learning_rate"].split(",")))
        l_dropout = list(map(float, config["dropout"].split(",")))
        l_spatial_dropout = list(map(float, config["spatial_dropout"].split(",")))
        l_recurrent_dropout = list(map(float, config["recurrent_dropout"].split(",")))

        optimize_n_epochs = int(config["optimize_n_epochs"])'''

        # get dimensions
        dimensions = len(reverse_dictionary) + 1
        best_model_params = dict()

        search_space = {
            'n_estimators': hp.choice('n_estimators', range(10, 20))
            'n_jobs': 
        }

        def classifier(params):
            rf_classifier = RandomForestClassifier(**params)
            trained_clf = rf_classifier.fit(train_data, train_labels)
            y_pred = trained_clf.predict(test_data)
            return utils.compute_weighted_loss(test_labels, y_pred, class_weights), trained_clf

        def optimise_clf(params):
            loss, model = classifier(params)
            return {'loss': loss, 'status': STATUS_OK, 'model': model}

        # minimize the objective function using the set of parameters above
        trials = Trials()
        learned_params = fmin(optimise_clf, search_space, trials=trials, algo=tpe.suggest, max_evals=int(config["max_evals"]))
        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        # set the best params with respective values
        for item in learned_params:
            item_val = learned_params[item]
            best_model_params[item] = item_val
        print(best_model_params)
        return best_model_params, best_model
