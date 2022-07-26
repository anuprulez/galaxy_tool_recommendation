import os
import numpy as np
import json
import h5py
import random
from numpy.random import choice
import matplotlib.pyplot as plt

import tensorflow as tf
#from tensorflow.compat.v2 import backend as K
from tensorflow.keras import backend
#crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


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


def save_h5_data(inp, tar, filename):
    hf_file = h5py.File(filename, 'w')
    hf_file.create_dataset("input", data=inp)
    hf_file.create_dataset("target", data=tar)
    hf_file.close()


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
        expanded_weights = tf.expand_dims(weight_values, axis=-1)
        bce = backend.binary_crossentropy(y_true, y_pred)
        return backend.dot(bce, expanded_weights)
    return weighted_binary_crossentropy


def verify_oversampling_freq(oversampled_tr_data, rev_dict):
    """
    Compute the frequency of tool sequences after oversampling
    """
    freq_dict = dict()
    freq_dict_names = dict()
    for tr_data in oversampled_tr_data:
        t_pos = np.where(tr_data > 0)[0]
        last_tool_id = str(int(tr_data[t_pos[-1]])) #str(int(tr_data[-1]))
        #print(tr_data, last_tool_id)
        if last_tool_id not in freq_dict:
            freq_dict[last_tool_id] = 0
            freq_dict_names[rev_dict[int(last_tool_id)]] = 0
        freq_dict[last_tool_id] += 1
        freq_dict_names[rev_dict[int(last_tool_id)]] += 1
    s_freq = dict(sorted(freq_dict_names.items(), key=lambda kv: kv[1], reverse=True))
    #print(s_freq)
    return s_freq


def collect_sampled_tool_freq(collected_dict, c_freq):
    for t in c_freq:
        if t not in collected_dict:
            collected_dict[t] = int(c_freq[t])
        else:
            collected_dict[t] += int(c_freq[t])
    return collected_dict


def save_data_as_dict(f_dict, r_dict, inp, tar, save_path):
    inp_tar = dict()
    for index, (i, t) in enumerate(zip(inp, tar)):
        i_pos = np.where(i > 0)[0]
        i_seq = ",".join([str(int(item)) for item in i[1:i_pos[-1] + 1]])
        t_pos = np.where(t > 0)[0]
        t_seq = ",".join([str(int(item)) for item in t[1:t_pos[-1] + 1]])
        if i_seq not in inp_tar:
            inp_tar[i_seq] = list()
        inp_tar[i_seq].append(t_seq)
    size = 0
    for item in inp_tar:
        size += len(inp_tar[item])
    print("Size saved file: ", size)
    write_file(save_path, inp_tar)
    

def read_train_test(datapath):
    file_obj = h5py.File(datapath, 'r')
    data_input = np.array(file_obj["input"])
    data_target = np.array(file_obj["target"])
    return data_input, data_target


def balanced_sample_generator(train_data, train_labels, batch_size, l_tool_tr_samples, reverse_dictionary):
    l_tool_frequencies = dict()
    while True:
        dimension = train_data.shape[1]
        n_classes = train_labels.shape[1]
        tool_ids = list(l_tool_tr_samples.keys())
        random.shuffle(tool_ids)
        generator_batch_data = np.zeros([batch_size, dimension])
        generator_batch_labels = np.zeros([batch_size, n_classes])
        generated_tool_ids = choice(tool_ids, batch_size)
        for i in range(batch_size):
            random_toolid = generated_tool_ids[i]
            sample_indices = l_tool_tr_samples[str(random_toolid)]
            random_index = random.sample(range(0, len(sample_indices)), 1)[0]
            random_tr_index = sample_indices[random_index]
            generator_batch_data[i] = train_data[random_tr_index]
            generator_batch_labels[i] = train_labels[random_tr_index]
        freq = verify_oversampling_freq(generator_batch_data, reverse_dictionary)
        l_tool_frequencies = collect_sampled_tool_freq(l_tool_frequencies, freq)
        write_file("log/data/generated_tool_frequencies.txt", l_tool_frequencies)
        yield generator_batch_data, generator_batch_labels


def compute_precision(model, x, y, reverse_data_dictionary, usage_scores, actual_classes_pos, topk, standard_conn, last_tool_id, lowest_tool_ids):
    """
    Compute absolute and compatible precision
    """
    pred_t_name = ""
    top_precision = 0.0
    mean_usage = 0.0
    usage_wt_score = list()
    pub_precision = 0.0
    lowest_pub_prec = 0.0
    lowest_norm_prec = 0.0
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
                # count precision only when there is actually true published tools
                if last_tool_id in lowest_tool_ids:
                    lowest_pub_prec = 1.0
                else:
                    lowest_pub_prec = np.nan
                if standard_topk_prediction_pos in usage_scores:
                    usage_wt_score.append(np.log(usage_scores[standard_topk_prediction_pos] + 1.0))
        else:
            # count precision only when there is actually true published tools
            # else set to np.nan. Set to 0 only when there is wrong prediction
            pub_precision = np.nan
            lowest_pub_prec = np.nan
    # compute scores for normal recommendations
    if normal_topk_prediction_pos in reverse_data_dictionary:
        pred_t_name = reverse_data_dictionary[int(normal_topk_prediction_pos)]
        if pred_t_name in actual_next_tool_names:
            if normal_topk_prediction_pos in usage_scores:
                usage_wt_score.append(np.log(usage_scores[normal_topk_prediction_pos] + 1.0))
            top_precision = 1.0
            if last_tool_id in lowest_tool_ids:
                lowest_norm_prec = 1.0
            else:
                lowest_norm_prec = np.nan
    if len(usage_wt_score) > 0:
        mean_usage = np.mean(usage_wt_score)
    return mean_usage, top_precision, pub_precision, lowest_pub_prec, lowest_norm_prec


def get_lowest_tools(l_tool_freq, fraction=0.25):
    l_tool_freq = dict(sorted(l_tool_freq.items(), key=lambda kv: kv[1], reverse=True))
    tool_ids = list(l_tool_freq.keys())
    lowest_ids = tool_ids[-int(len(tool_ids) * fraction):]
    return lowest_ids


def verify_model(model, x, y, reverse_data_dictionary, usage_scores, standard_conn, lowest_tool_ids, topk_list=[1, 2, 3]):
    """
    Verify the model on test data
    """
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(y))
    size = y.shape[0]
    precision = np.zeros([len(y), len(topk_list)])
    usage_weights = np.zeros([len(y), len(topk_list)])
    epo_pub_prec = np.zeros([len(y), len(topk_list)])
    epo_lowest_tools_pub_prec = list()
    epo_lowest_tools_norm_prec = list()
    lowest_counter = 0
    # loop over all the test samples and find prediction precision
    for i in range(size):
        lowest_pub_topk = list()
        lowest_norm_topk = list()
        actual_classes_pos = np.where(y[i] > 0)[0]
        test_sample = x[i, :]
        last_tool_id = str(int(test_sample[-1]))
        for index, abs_topk in enumerate(topk_list):
            usg_wt_score, absolute_precision, pub_prec, lowest_p_prec, lowest_n_prec = compute_precision(model, test_sample, y, reverse_data_dictionary, usage_scores, actual_classes_pos, abs_topk, standard_conn, last_tool_id, lowest_tool_ids)
            precision[i][index] = absolute_precision
            usage_weights[i][index] = usg_wt_score
            epo_pub_prec[i][index] = pub_prec
            lowest_pub_topk.append(lowest_p_prec)
            lowest_norm_topk.append(lowest_n_prec)
        epo_lowest_tools_pub_prec.append(lowest_pub_topk)
        epo_lowest_tools_norm_prec.append(lowest_norm_topk)
        if last_tool_id in lowest_tool_ids:
            lowest_counter += 1
    mean_precision = np.mean(precision, axis=0)
    mean_usage = np.mean(usage_weights, axis=0)
    mean_pub_prec = np.nanmean(epo_pub_prec, axis=0)
    mean_lowest_pub_prec = np.nanmean(epo_lowest_tools_pub_prec, axis=0)
    mean_lowest_norm_prec = np.nanmean(epo_lowest_tools_norm_prec, axis=0)
    return mean_usage, mean_precision, mean_pub_prec, mean_lowest_pub_prec, mean_lowest_norm_prec, lowest_counter


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
    if "lowest_pub_precision" in results:
        np.savetxt("data/lowest_pub_precision.txt", results["lowest_pub_precision"])
    if "lowest_norm_precision" in results:
        np.savetxt("data/lowest_norm_precision.txt", results["lowest_norm_precision"])


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
