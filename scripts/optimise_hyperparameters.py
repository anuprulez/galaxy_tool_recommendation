"""
Find the optimal combination of hyperparameters
"""

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from  sklearn.ensemble import ExtraTreesClassifier

import utils


class HyperparameterOptimisation:

    def __init__(self):
        """ Init method. """

    def train_model(self, config, reverse_dictionary, train_data, train_labels, test_data, test_labels, tool_tr_samples, class_weights):
        """
        Train a model and report accuracy
        """
        l_estimators = list(map(int, config["n_estimators"].split(",")))
        l_criterion = config["criterion"].split(",")
        l_max_depth = list(map(int, config["max_depth"].split(",")))
        l_min_samples_split = list(map(float, config["min_samples_split"].split(",")))
        l_max_features = config["max_features"].split(",")
        l_max_features.append(None)
        l_bootstrap = [True, False]

        dimensions = len(reverse_dictionary) + 1
        best_model_params = dict()

        search_space = {
            'n_estimators': hp.choice('n_estimators', range(l_estimators[0], l_estimators[1])),
            'max_depth': hp.choice('max_depth', range(l_max_depth[0], l_max_depth[1])),
            'min_samples_split': hp.loguniform("min_samples_split", np.log(l_min_samples_split[0]), np.log(l_min_samples_split[1])),
            'criterion': hp.choice('criterion', l_criterion),
            'max_features': hp.choice('max_features', l_max_features),
            'bootstrap': hp.choice('bootstrap', l_bootstrap),
            'n_jobs': int(config["num_cpus"])
        }

        def classifier(params):
            rf_classifier = ExtraTreesClassifier(**params)
            tr_data, tr_labels = utils.balanced_sample_generator(train_data, train_labels, train_data.shape[0], tool_tr_samples, reverse_dictionary)
            trained_clf = rf_classifier.fit(tr_data, tr_labels)
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
            if item == "criterion":
                val = l_criterion[learned_params[item]]
            elif item == "bootstrap":
                val = l_bootstrap[learned_params[item]]
            elif item == "max_features":
                val = l_max_features[learned_params[item]]
            else:
                val = learned_params[item]
            best_model_params[item] = val
        return best_model_params, best_model
