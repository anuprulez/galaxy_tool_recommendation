"""
Prepare the workflow paths to be used by downstream
machine learning algorithm. The paths are divided
into the test and training sets
"""

import os
import collections
import numpy as np
import random

import predict_tool_usage
import utils

main_path = os.getcwd()


class PrepareData:

    def __init__(self, max_seq_length, test_data_share, train_size):
        """ Init method. """
        self.max_tool_sequence_len = max_seq_length
        self.test_share = test_data_share
        self.train_size = train_size

    def process_workflow_paths(self, workflow_paths):
        """
        Get all the tools and complete set of individual paths for each workflow
        """
        tokens = list()
        raw_paths = workflow_paths
        raw_paths = [x.replace("\n", '') for x in raw_paths]
        for item in raw_paths:
            split_items = item.split(",")
            for token in split_items:
                if token is not "":
                    tokens.append(token)
        tokens = list(set(tokens))
        tokens = np.array(tokens)
        tokens = np.reshape(tokens, [-1, ])
        return tokens, raw_paths

    def create_new_dict(self, new_data_dict):
        """
        Create new data dictionary
        """
        reverse_dict = dict((v, k) for k, v in new_data_dict.items())
        return new_data_dict, reverse_dict

    def assemble_dictionary(self, new_data_dict, old_data_dictionary={}):
        """
        Create/update tools indices in the forward and backward dictionary
        """
        new_data_dict, reverse_dict = self.create_new_dict(new_data_dict)
        return new_data_dict, reverse_dict

    def create_data_dictionary(self, words, old_data_dictionary={}):
        """
        Create two dictionaries having tools names and their indexes
        """
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary) + 1
        dictionary, reverse_dictionary = self.assemble_dictionary(dictionary, old_data_dictionary)
        return dictionary, reverse_dictionary

    def decompose_paths(self, paths, dictionary):
        """
        Decompose the paths to variable length sub-paths keeping the first tool fixed
        """
        sub_paths_pos = list()
        for index, item in enumerate(paths):
            tools = item.split(",")
            len_tools = len(tools)
            if len_tools <= self.max_tool_sequence_len:
                for window in range(1, len_tools):
                    sequence = tools[0: window + 1]
                    tools_pos = [str(dictionary[str(tool_item)]) for tool_item in sequence]
                    if len(tools_pos) > 1:
                        sub_paths_pos.append(",".join(tools_pos))
        sub_paths_pos = list(set(sub_paths_pos))
        return sub_paths_pos

    def prepare_paths_labels_dictionary(self, dictionary, reverse_dictionary, paths, compatible_next_tools):
        """
        Create a dictionary of sequences with their labels for training and test paths
        """
        paths_labels = dict()
        random.shuffle(paths)
        for item in paths:
            if item and item not in "":
                tools = item.split(",")
                label = tools[-1]
                train_tools = tools[:len(tools) - 1]
                last_but_one_name = reverse_dictionary[int(train_tools[-1])]
                try:
                    compatible_tools = compatible_next_tools[last_but_one_name].split(",")
                except Exception:
                    continue
                if len(compatible_tools) > 0:
                    compatible_tools_ids = [str(dictionary[x]) for x in compatible_tools]
                    compatible_tools_ids.append(label)
                    composite_labels = ",".join(compatible_tools_ids)
                train_tools = ",".join(train_tools)
                if train_tools in paths_labels:
                    paths_labels[train_tools] += "," + composite_labels
                else:
                    paths_labels[train_tools] = composite_labels
        for item in paths_labels:
            paths_labels[item] = ",".join(list(set(paths_labels[item].split(","))))
        return paths_labels

    def pad_test_paths(self, paths_dictionary, num_classes):
        """
        Add padding to the tools sequences and create multi-hot encoded labels
        """
        size_data = len(paths_dictionary)
        data_mat = np.zeros([size_data, self.max_tool_sequence_len])
        label_mat = np.zeros([size_data, num_classes + 1])
        train_counter = 0
        for train_seq, train_label in list(paths_dictionary.items()):
            positions = train_seq.split(",")
            start_pos = self.max_tool_sequence_len - len(positions)
            for id_pos, pos in enumerate(positions):
                data_mat[train_counter][start_pos + id_pos] = int(pos)
            for label_item in train_label.split(","):
                label_mat[train_counter][int(label_item)] = 1.0
            train_counter += 1
        return data_mat, label_mat

    def pad_paths(self, paths_dictionary, num_classes, standard_connections, reverse_dictionary):
        """
        Add padding to the tools sequences and create multi-hot encoded labels
        """
        size_data = len(paths_dictionary)
        data_mat = np.zeros([size_data, self.max_tool_sequence_len])
        label_mat = np.zeros([size_data, 2 * (num_classes + 1)])
        pos_flag = 1.0
        train_counter = 0
        for train_seq, train_label in list(paths_dictionary.items()):
            pub_connections = list()
            positions = train_seq.split(",")
            last_tool_id = positions[-1]
            last_tool_name = reverse_dictionary[int(last_tool_id)]
            start_pos = self.max_tool_sequence_len - len(positions)
            for id_pos, pos in enumerate(positions):
                data_mat[train_counter][start_pos + id_pos] = int(pos)
            if last_tool_name in standard_connections:
                pub_connections = standard_connections[last_tool_name]
            for label_item in train_label.split(","):
                label_pos = int(label_item)
                label_row = label_mat[train_counter]
                if reverse_dictionary[label_pos] in pub_connections:
                    label_row[label_pos] = pos_flag
                else:
                    label_row[label_pos + num_classes + 1] = pos_flag
            train_counter += 1
        return data_mat, label_mat

    def split_test_train_data(self, multilabels_paths):
        """
        Split into test and train data randomly for each run
        """
        train_dict = dict()
        test_dict = dict()
        all_paths = multilabels_paths.keys()
        random.shuffle(list(all_paths))
        split_number = int(self.test_share * len(all_paths))
        for index, path in enumerate(list(all_paths)):
            if index < split_number:
                test_dict[path] = multilabels_paths[path]
            else:
                train_dict[path] = multilabels_paths[path]
        return train_dict, test_dict

    def get_predicted_usage(self, data_dictionary, predicted_usage):
        """
        Get predicted usage for tools
        """
        usage = dict()
        epsilon = 0.0
        # index 0 does not belong to any tool
        usage[0] = epsilon
        for k, v in data_dictionary.items():
            try:
                usg = predicted_usage[k]
                if usg < epsilon:
                    usg = epsilon
                usage[v] = usg
            except Exception:
                usage[v] = epsilon
                continue
        return usage
        
    def compute_sample_weight(self, train_data, inv_last_tool_freq):
        inv_train_sample_weight = list()
        for ts in train_data:
            l_tool = int(ts[-1])
            sample_wt = inv_last_tool_freq[str(l_tool)]
            inv_train_sample_weight.append(sample_wt)
        return inv_train_sample_weight

    def assign_class_weights(self, n_classes, predicted_usage):
        """
        Compute class weights using usage
        """
        class_weights = dict()
        class_weights[str(0)] = 0.0
        for key in range(1, n_classes + 1):
            u_score = predicted_usage[key]
            if u_score < 1.0:
                u_score += 1.0
            class_weights[key] = np.round(np.log(u_score), 6)
        return class_weights

    def get_sample_weights(self, train_data, reverse_dictionary, paths_frequency):
        """
        Compute the frequency of paths in training data
        """
        path_weights = np.zeros(len(train_data))
        for path_index, path in enumerate(train_data):
            sample_pos = np.where(path > 0)[0]
            sample_tool_pos = path[sample_pos[0]:]
            path_name = ",".join([reverse_dictionary[int(tool_pos)] for tool_pos in sample_tool_pos])
            try:
                path_weights[path_index] = int(paths_frequency[path_name])
            except Exception:
                path_weights[path_index] = 1
        return path_weights
        
    def get_train_last_tool_freq(self, train_paths, reverse_dictionary):
        last_tool_freq = dict()
        inv_freq = dict()
        for path in train_paths:
            last_tool = path.split(",")[-1]
            if last_tool not in last_tool_freq:
                last_tool_freq[last_tool] = 0
            last_tool_freq[last_tool] += 1
        max_freq = max(last_tool_freq.values())
        for t in last_tool_freq:
            inv_freq[t] = int(np.round(max_freq / float(last_tool_freq[t]), 0))
        utils.write_file("data/last_tool_freq.txt", last_tool_freq)
        utils.write_file("data/inverse_last_tool_freq.txt", inv_freq)
        return last_tool_freq, inv_freq
        
    def oversample_training(self, train_data, train_labels, inv_freq, actual_freq):
        oversampled_train_data = list()
        oversampled_train_labels = list()
        freq_dict = dict()
        for index, tr_sample in enumerate(train_data):
            last_tool_id = str(int(tr_sample[-1]))
            n_repeat = inv_freq[last_tool_id]
            label_vec = train_labels[index]
            oversampled_tr = np.array([tr_sample] * n_repeat)
            oversampled_label = np.array([label_vec] * n_repeat)
            oversampled_train_data.extend(oversampled_tr.tolist())
            oversampled_train_labels.extend(oversampled_label.tolist())
        assert len(oversampled_train_data) == len(oversampled_train_labels)
        return oversampled_train_data, oversampled_train_labels
        
    def verify_oversampling_freq(self, oversampled_tr_data):
        freq_dict = dict()
        for tr_data in oversampled_tr_data:
            last_tool_id = str(int(tr_data[-1]))
            if last_tool_id not in freq_dict:
                freq_dict[last_tool_id] = 0
            freq_dict[last_tool_id] += 1
        utils.write_file("data/oversampled_last_tool_freq.txt", freq_dict)

    def sample_train_data(self, oversampled_train_data, oversampled_train_labels, num_classes):
        # randomize oversampled data
        n_oversample = len(oversampled_train_data)
        random_tr_data = list()
        random_tr_label = list()
        random_oversample_pos = random.sample(range(0, n_oversample), n_oversample)
        for pos in random_oversample_pos:
            random_tr_data.append(oversampled_train_data[pos])
            random_tr_label.append(oversampled_train_labels[pos])
        # create a undersampled train data from randomized oversampled data 
        random_pos = random.sample(range(0, len(random_tr_data)), self.train_size)
        train_data = np.zeros([self.train_size, self.max_tool_sequence_len])
        train_labels = np.zeros([self.train_size, 2 * (num_classes + 1)])
        
        for index, pos in enumerate(random_pos):
            train_data[index] = random_tr_data[pos]
            train_labels[index] = random_tr_label[pos]
        self.verify_oversampling_freq(train_data)
        return train_data, train_labels

    def get_data_labels_matrices(self, workflow_paths, tool_usage_path, cutoff_date, compatible_next_tools, standard_connections, old_data_dictionary={}):
        """
        Convert the training and test paths into corresponding numpy matrices
        """
        processed_data, raw_paths = self.process_workflow_paths(workflow_paths)
        dictionary, rev_dict = self.create_data_dictionary(processed_data, old_data_dictionary)
        num_classes = len(dictionary)

        print("Raw paths: %d" % len(raw_paths))
        random.shuffle(raw_paths)

        print("Decomposing paths...")
        all_unique_paths = self.decompose_paths(raw_paths, dictionary)
        random.shuffle(all_unique_paths)

        print("Creating dictionaries...")
        multilabels_paths = self.prepare_paths_labels_dictionary(dictionary, rev_dict, all_unique_paths, compatible_next_tools)

        print("Complete data: %d" % len(multilabels_paths))
        train_paths_dict, test_paths_dict = self.split_test_train_data(multilabels_paths)
        
        # get sample frequency
        l_tool_freq, inv_last_tool_freq = self.get_train_last_tool_freq(train_paths_dict, rev_dict)
        
        print("Train data: %d" % len(train_paths_dict))
        print("Test data: %d" % len(test_paths_dict))

        test_data, test_labels = self.pad_paths(test_paths_dict, num_classes, standard_connections, rev_dict)
        train_data, train_labels = self.pad_paths(train_paths_dict, num_classes, standard_connections, rev_dict)
        
        oversampled_train_data, oversampled_train_labels = self.oversample_training(train_data, train_labels, inv_last_tool_freq, l_tool_freq)
        
        self.verify_oversampling_freq(oversampled_train_data)
        
        train_data, train_labels = self.sample_train_data(oversampled_train_data, oversampled_train_labels, num_classes)
        
        print("Train data after oversampling: %d" % len(train_data))
        
        #import sys
        #sys.exit()

        # Predict tools usage
        print("Predicting tools' usage...")
        usage_pred = predict_tool_usage.ToolPopularity()
        usage = usage_pred.extract_tool_usage(tool_usage_path, cutoff_date, dictionary)
        tool_usage_prediction = usage_pred.get_pupularity_prediction(usage)
        t_pred_usage = self.get_predicted_usage(dictionary, tool_usage_prediction)

        # get class weights using the predicted usage for each tool
        class_weights = self.assign_class_weights(num_classes, t_pred_usage)

        return train_data, train_labels, test_data, test_labels, dictionary, rev_dict, class_weights, t_pred_usage, l_tool_freq
