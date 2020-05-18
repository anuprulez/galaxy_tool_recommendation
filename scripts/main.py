"""
Fetch recommended tools in Galaxy workflows
using statistical model
"""

import argparse
import time
import h5py
import json

import extract_workflow_connections
import prepare_data
import utils


class PredictTool:

    def __init__(self):
        """ Init method. """

    def create_model(self, multilabels_paths, data_dictionary, class_weights, compatible_next_tools, model_path, standard_connections):
        data = {
            "multilabels_paths": multilabels_paths,
            "data_dictionary": data_dictionary,
            "compatible_next_tools": compatible_next_tools,
            "class_weights": class_weights,
            "standard_connections": standard_connections
        }
        utils.save_model(data, model_path)

    def predict_tools(self, model_path, test_path="Cut1"):
        dict_paths, d_dict, c_tools, class_weights = utils.load_model(model_path)
        rev_dict = dict((str(v), k) for k, v in d_dict.items())
        p_num = list()
        for t in test_path.split(","):
            p_num.append(str(d_dict[t]))
        p_num = ",".join(p_num)
        predicted_tools = list()
        for k in dict_paths:
            if k == p_num:
                predicted_tools = dict_paths[k].split(",")
                break
        pred_names = list()
        for tool in predicted_tools:
            pred_names.append(rev_dict[tool])
        assert len(pred_names) > 0
        print("Test path: %s" % test_path)
        print("Predicted tools: %s" % ",".join(pred_names))


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-wf", "--workflow_file", required=True, help="workflows tabular file")
    arg_parser.add_argument("-tu", "--tool_usage_file", required=True, help="tool usage file")
    arg_parser.add_argument("-om", "--output_model", required=True, help="model file")
    # data parameters
    arg_parser.add_argument("-cd", "--cutoff_date", required=True, help="earliest date for taking tool usage")
    arg_parser.add_argument("-pl", "--maximum_path_length", required=True, help="maximum length of tool path")

    # get argument values
    args = vars(arg_parser.parse_args())
    tool_usage_path = args["tool_usage_file"]
    workflows_path = args["workflow_file"]
    cutoff_date = args["cutoff_date"]
    maximum_path_length = int(args["maximum_path_length"])
    model_path = args["output_model"]

    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    workflow_paths, compatible_next_tools, standard_connections = connections.read_tabular_file(workflows_path)
    # Process the paths from workflows
    print("Dividing data...")
    data = prepare_data.PrepareData(maximum_path_length)
    multilabels_paths, data_dictionary, reverse_dictionary, class_weights, usage_pred = data.get_data_labels_matrices(workflow_paths, tool_usage_path, cutoff_date, compatible_next_tools)
    # create model and test predicted tools
    predict_tool = PredictTool()
    predict_tool.create_model(multilabels_paths, data_dictionary, class_weights, compatible_next_tools, model_path, standard_connections)
    print()
    predict_tool.predict_tools(model_path)
    end_time = time.time()
    print()
    print("Program finished in %s seconds" % str(end_time - start_time))
