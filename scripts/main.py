"""
Predict next tools in the Galaxy workflows
using machine learning (recurrent neural network)
"""

import numpy as np
import argparse
import time

# machine learning library
import tensorflow as tf
from keras import backend as K
import keras.callbacks as callbacks

import extract_workflow_connections
import prepare_data
import optimise_hyperparameters
import utils


class PredictTool:

    def __init__(self):
        """ Init method. """

    def find_train_best_network(self, network_config, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights, usage_pred, standard_connections, tool_freq, tool_tr_samples):
        """
        Define recurrent neural network and train sequential data
        """
        print("Start hyperparameter optimisation...")
        train_performance = dict()
        hyper_opt = optimise_hyperparameters.HyperparameterOptimisation()
        best_params, best_model = hyper_opt.train_model(network_config, reverse_dictionary, train_data, train_labels, test_data, test_labels, tool_tr_samples, class_weights)
        evaluate_m = EvaluateModel(test_data, test_labels, reverse_dictionary, usage_pred, standard_connections, best_model)
        usage_weights, precision, published_precision = evaluate_m.evaluate_test()

        train_performance["precision"] = precision
        train_performance["usage_weights"] = usage_weights
        train_performance["published_precision"] = published_precision
        train_performance["model"] = best_model
        train_performance["best_parameters"] = best_params
        return train_performance


class EvaluateModel():
    def __init__(self, test_data, test_labels, reverse_data_dictionary, usg_scores, standard_connections, best_model):
        self.test_data = test_data
        self.test_labels = test_labels
        self.reverse_data_dictionary = reverse_data_dictionary
        self.precision = list()
        self.usage_weights = list()
        self.published_precision = list()
        self.pred_usage_scores = usg_scores
        self.standard_connections = standard_connections
        self.model = best_model

    def evaluate_test(self):
        """
        Compute absolute and compatible precision for test data
        """
        if len(self.test_data) > 0:
            usage_weights, precision, precision_pub = utils.verify_model(self.model, self.test_data, self.test_labels, self.reverse_data_dictionary, self.pred_usage_scores, self.standard_connections)
            self.precision.append(precision)
            self.usage_weights.append(usage_weights)
            self.published_precision.append(precision_pub)
            print("Usage weights: %s" % (usage_weights))
            print("Normal precision: %s" % (precision))
            print("Published precision: %s" % (precision_pub))
            return self.usage_weights, self.precision, self.published_precision


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-wf", "--workflow_file", required=True, help="workflows tabular file")
    arg_parser.add_argument("-tu", "--tool_usage_file", required=True, help="tool usage file")
    arg_parser.add_argument("-om", "--output_model", required=True, help="model file")
    arg_parser.add_argument("-tm", "--trained_model", required=True, help="trained model file")
    # data parameters
    arg_parser.add_argument("-cd", "--cutoff_date", required=True, help="earliest date for taking tool usage")
    arg_parser.add_argument("-pl", "--maximum_path_length", required=True, help="maximum length of tool path")
    arg_parser.add_argument("-me", "--max_evals", required=True, help="maximum number of configuration evaluations")
    arg_parser.add_argument("-ts", "--test_share", required=True, help="share of data to be used for testing")
    # neural network parameters
    arg_parser.add_argument("-ne", "--num_estimators", required=True, help="number of estimators")
    arg_parser.add_argument("-ct", "--criterion", required=True, help="function to measure the quality of a split")
    arg_parser.add_argument("-mss", "--min_samples_split", required=True, help="size of the fixed vector learned for each tool")
    arg_parser.add_argument("-md", "--max_depth", required=True, help="maximum depth of the tree")
    arg_parser.add_argument("-mf", "--max_features", required=True, help="number of features to consider when looking for the best split")
    arg_parser.add_argument("-cpus", "--num_cpus", required=True, help="number of cpus for parallelism")

    # get argument values
    args = vars(arg_parser.parse_args())
    tool_usage_path = args["tool_usage_file"]
    workflows_path = args["workflow_file"]
    cutoff_date = args["cutoff_date"]
    maximum_path_length = int(args["maximum_path_length"])
    output_model_path = args["output_model"]
    trained_model_path = args["trained_model"]
    max_evals = int(args["max_evals"])
    test_share = float(args["test_share"])

    n_estimators = args["num_estimators"]
    criterion = args["criterion"]
    min_samples_split = args["min_samples_split"]
    max_depth = args["max_depth"]
    max_features = args["max_features"]
    num_cpus = int(args["num_cpus"])

    config = {
        'cutoff_date': cutoff_date,
        'maximum_path_length': maximum_path_length,
        'max_evals': max_evals,
        'test_share': test_share,
        'n_estimators': n_estimators,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'max_depth': max_depth,
        'max_features': max_features,
        'num_cpus': num_cpus
    }

    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    workflow_paths, compatible_next_tools, standard_connections = connections.read_tabular_file(workflows_path)
    # Process the paths from workflows
    print("Dividing data...")
    data = prepare_data.PrepareData(maximum_path_length, test_share)
    train_data, train_labels, test_data, test_labels, data_dictionary, reverse_dictionary, class_weights, usage_pred, train_tool_freq, tool_tr_samples = data.get_data_labels_matrices(workflow_paths, tool_usage_path, cutoff_date, compatible_next_tools, standard_connections)
    # find the best model and start training
    predict_tool = PredictTool()
    # start training with weighted classes
    print("Training with weighted classes and samples ...")
    results_weighted = predict_tool.find_train_best_network(config, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights, usage_pred, standard_connections, train_tool_freq, tool_tr_samples)
    print()
    print("Best parameters \n")
    print(results_weighted["best_parameters"])
    print()
    utils.write_file("data/train_last_tools.txt", train_tool_freq)
    utils.save_model(results_weighted, data_dictionary, compatible_next_tools, output_model_path, trained_model_path, class_weights, standard_connections)
    end_time = time.time()
    print()
    print("Program finished in %s seconds" % str(end_time - start_time))
