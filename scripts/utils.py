import os
import numpy as np
import json
import h5py

from tensorflow.keras import backend as K


def read_file(file_path):
    """
    Read a file
    """
    with open(file_path, "r") as json_file:
        file_content = json.loads(json_file.read())
    return file_content


def write_file(file_path, content):
    """
    Write a file
    """
    remove_file(file_path)
    with open(file_path, "w") as json_file:
        json_file.write(json.dumps(content))


def save_processed_workflows(file_path, unique_paths):
    workflow_paths_unique = ""
    for path in unique_paths:
        workflow_paths_unique += path + "\n"
    with open(file_path, "w") as workflows_file:
        workflows_file.write(workflow_paths_unique)


def format_tool_id(tool_link):
    """
    Extract tool id from tool link
    """
    tool_id_split = tool_link.split("/")
    tool_id = tool_id_split[-2] if len(tool_id_split) > 1 else tool_link
    return tool_id


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def weighted_loss(class_weights):
    """
    Create a weighted loss function. Penalise the misclassification
    of classes more with the higher usage
    """
    weight_values = list(class_weights.values())

    def weighted_binary_crossentropy(y_true, y_pred):
        # add another dimension to compute dot product
        expanded_weights = K.expand_dims(weight_values, axis=-1)
        return K.dot(K.binary_crossentropy(y_true, y_pred), expanded_weights)
    return weighted_binary_crossentropy


def compute_precision(model, x, reverse_data_dictionary, usage_scores, actual_classes_pos, topk):
    """
    Compute absolute and compatible precision
    """
    absolute_precision = 0.0
    test_sample = np.reshape(x, (1, len(x)))

    # predict next tools for a test path
    prediction = model.predict(test_sample, verbose=0)

    nw_dimension = prediction.shape[1]

    # remove the 0th position as there is no tool at this index
    prediction = np.reshape(prediction, (nw_dimension,))

    prediction_pos = np.argsort(prediction, axis=-1)
    topk_prediction_pos = prediction_pos[-topk:]

    # remove the wrong tool position from the predicted list of tool positions
    topk_prediction_pos = [x for x in topk_prediction_pos if x > 0]

    # read tool names using reverse dictionary
    #actual_next_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in actual_classes_pos]
    #top_predicted_next_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in topk_prediction_pos]

    # compute the class weights of predicted tools
    mean_usg_score = 0
    usg_wt_scores = list()
    for t_id in topk_prediction_pos:
        #t_name = reverse_data_dictionary[int(t_id)]
        if t_id in usage_scores and t_id in actual_classes_pos:
            usg_wt_scores.append(np.log(usage_scores[t_id] + 1.0))
    if len(usg_wt_scores) > 0:
            mean_usg_score = np.sum(usg_wt_scores) / float(topk)
    false_positives = [t_id for t_id in topk_prediction_pos if t_id not in actual_classes_pos]
    absolute_precision = 1 - (len(false_positives) / float(topk))
    return mean_usg_score, absolute_precision


def verify_model(model, x, y, reverse_data_dictionary, usage_scores, topk_list=[1, 2, 3]):
    """
    Verify the model on test data
    """
    print()
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(y))
    size = y.shape[0]
    precision = np.zeros([len(y), len(topk_list)])
    usage_weights = np.zeros([len(y), len(topk_list)])
    # loop over all the test samples and find prediction precision
    for i in range(size):
        actual_classes_pos = np.where(y[i] > 0)[0]
        for index, abs_topk in enumerate(topk_list):
            abs_mean_usg_score, absolute_precision = compute_precision(model, x[i, :], reverse_data_dictionary, usage_scores, actual_classes_pos, abs_topk)
            precision[i][index] = absolute_precision
            usage_weights[i][index] = abs_mean_usg_score
    mean_precision = np.mean(precision, axis=0)
    mean_usage = np.mean(usage_weights, axis=0)
    return mean_precision, mean_usage


def set_trained_model(dump_file, model_values):
    """
    Append helper dictionaries to trained model h5 file
    """
    hf_file = h5py.File(dump_file, 'a')
    for key in model_values:
        value = model_values[key]
        hf_file.create_dataset(key, data=json.dumps(value))   
    hf_file.close()


def save_model(results, data_dictionary, compatible_next_tools, trained_model_path, class_weights, max_path_len):
    # save files
    trained_model = results["model"]
    trained_model.save(trained_model_path)
    parameters = results["best_parameters"]
    parameters["max_len"] = max_path_len
    parameters["dimensions"] = len(class_weights)
    parameters["class_weights"] = class_weights
    model_values = {
        'data_dictionary': data_dictionary,
        'parameters': parameters,
        "compatible_tools": compatible_next_tools,
    }
    set_trained_model(trained_model_path, model_values)
