import os
import numpy as np
import json
import h5py

from keras import backend as K


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


def set_trained_model(dump_file, model_values):
    """
    Create an h5 file with the trained weights and associated dicts
    """
    hf_file = h5py.File(dump_file, 'w')
    for key in model_values:
        value = model_values[key]
        if key == 'model_weights':
            for idx, item in enumerate(value):
                w_key = "weight_" + str(idx)
                if w_key in hf_file:
                    hf_file.modify(w_key, item)
                else:
                    hf_file.create_dataset(w_key, data=item)
        else:
            if key in hf_file:
                hf_file.modify(key, json.dumps(value))
            else:
                hf_file.create_dataset(key, data=json.dumps(value))
    hf_file.close()


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def weighted_loss(class_weights):
    """
    Create a weighted loss function. Penalise the misclassification
    of classes more with the higher usage
    """
    weight_values = list(class_weights.values())
    weight_values.extend(weight_values)
    def weighted_binary_crossentropy(y_true, y_pred):
        # add another dimension to compute dot product
        expanded_weights = K.expand_dims(weight_values, axis=-1)
        return K.dot(K.binary_crossentropy(y_true, y_pred), expanded_weights)
    return weighted_binary_crossentropy


def compute_precision(model, x, y, reverse_data_dictionary, usage_scores, actual_classes_pos, topk, standard_conn):
    """
    Compute absolute and compatible precision
    """
    pred_t_name = ""
    top_precision = 0.0
    mean_usage = 0.0
    usage_wt_score = list()
    pub_precision = 0.0
    pub_tools = list()
    actual_next_tool_names = list()
    test_sample = np.reshape(x, (1, len(x)))

    # predict next tools for a test path
    prediction = model.predict(test_sample, verbose=0)

    # divide the predicted vector into two halves - one for published and
    # another for normal workflows
    nw_dimension = prediction.shape[1]
    half_len = int(nw_dimension / 2)

    # predict tools
    prediction = np.reshape(prediction, (nw_dimension,))
    
    # get predictions of tools from published workflows
    standard_pred = prediction[:half_len]
    # get predictions of tools from normal workflows
    normal_pred = prediction[half_len:]

    standard_prediction_pos = np.argsort(standard_pred, axis=-1)
    standard_topk_prediction_pos = standard_prediction_pos[-topk]

    normal_prediction_pos = np.argsort(normal_pred, axis=-1)
    normal_topk_prediction_pos = normal_prediction_pos[-topk]

    # get true tools names
    for a_t_pos in actual_classes_pos:
        if a_t_pos > half_len:
            t_name = reverse_data_dictionary[int(a_t_pos - half_len)]
        else:
            t_name = reverse_data_dictionary[int(a_t_pos)]
        actual_next_tool_names.append(t_name)
    last_tool_name = reverse_data_dictionary[x[-1]]
    # compute scores for published recommendations
    if standard_topk_prediction_pos in reverse_data_dictionary:
        pred_t_name = reverse_data_dictionary[int(standard_topk_prediction_pos)]
        if last_tool_name in standard_conn:
            pub_tools = standard_conn[last_tool_name]
        if pred_t_name in pub_tools:
            pub_precision = 1.0
            if standard_topk_prediction_pos in usage_scores:
                usage_wt_score.append(np.log(usage_scores[standard_topk_prediction_pos] + 1.0))
    # compute scores for normal recommendations
    if normal_topk_prediction_pos in reverse_data_dictionary:
        pred_t_name = reverse_data_dictionary[int(normal_topk_prediction_pos)]
        if pred_t_name in actual_next_tool_names:
            if normal_topk_prediction_pos in usage_scores:
                usage_wt_score.append(np.log(usage_scores[normal_topk_prediction_pos] + 1.0))
            top_precision = 1.0
    if len(usage_wt_score) > 0:
        mean_usage = np.mean(usage_wt_score)
    return top_precision, mean_usage, pub_precision


def verify_model(model, x, y, reverse_data_dictionary, usage_scores, standard_conn, topk_list=[1, 2, 3]):
    """
    Verify the model on test data
    """
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(y))
    size = y.shape[0]
    precision = np.zeros([len(y), len(topk_list)])
    usage_weights = np.zeros([len(y), len(topk_list)])
    epo_pub_prec = np.zeros([len(y), len(topk_list)])
    # loop over all the test samples and find prediction precision
    for i in range(size):
        actual_classes_pos = np.where(y[i] > 0)[0]
        for index, abs_topk in enumerate(topk_list):
            absolute_precision, usg_wt_score, pub_prec = compute_precision(model, x[i, :], y, reverse_data_dictionary, usage_scores, actual_classes_pos, abs_topk, standard_conn)
            precision[i][index] = absolute_precision
            usage_weights[i][index] = usg_wt_score
            epo_pub_prec[i][index] = pub_prec
    mean_precision = np.mean(precision, axis=0)
    mean_usage = np.mean(usage_weights, axis=0)
    mean_pub_prec = np.mean(epo_pub_prec, axis=0)
    return mean_precision, mean_usage, mean_pub_prec


def save_results(results):
    if "validation_loss" in results:
        np.savetxt("data/validation_loss.txt", results["validation_loss"])
    if "train_loss" in results:
        np.savetxt("data/train_loss.txt", results["train_loss"])
    if "precision" in results:
        np.savetxt("data/precision.txt", results["precision"])
    if "usage_weights" in results:
        np.savetxt("data/usage_weights.txt", results["usage_weights"])
    if "published_precision" in results:
        np.savetxt("data/published_precision.txt", results["published_precision"])
    if "standard_connections" in results:
        write_file("data/standard_connections.txt", results["standard_connections"])


def save_model(results, data_dictionary, compatible_next_tools, trained_model_path, class_weights, standard_connections):
    # save files
    trained_model = results["model"]
    best_model_parameters = results["best_parameters"]
    model_config = trained_model.to_json()
    model_weights = trained_model.get_weights()
    model_values = {
        'data_dictionary': data_dictionary,
        'model_config': model_config,
        'best_parameters': best_model_parameters,
        'model_weights': model_weights,
        "compatible_tools": compatible_next_tools,
        "class_weights": class_weights,
        "standard_connections": standard_connections
    }
    set_trained_model(trained_model_path, model_values)
    results["standard_connections"] = standard_connections
    save_results(results)
