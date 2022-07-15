"""
Prepare the workflow paths to be used by downstream
machine learning algorithm. The paths are divided
into the test and training sets
"""

import os
import collections
import numpy as np
import random

from scripts import predict_tool_usage
from scripts import utils

main_path = os.getcwd()
start_token_name = "start_token"
index_start_token = 2


class PrepareData:

    def __init__(self, max_seq_length, test_data_share):
        """ Init method. """
        self.max_tool_sequence_len = max_seq_length
        self.test_share = test_data_share

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
        for index, (word, _) in enumerate(count):
            word = word.lstrip()
            word = word.rstrip()
            if index == 1:
                dictionary[word] = len(dictionary) + 2
                dictionary[start_token_name] = index_start_token
            else:
                dictionary[word] = len(dictionary) + 1
        #dictionary["start_token"] = index_start_token
        dictionary, reverse_dictionary = self.assemble_dictionary(dictionary, old_data_dictionary)
        print(dictionary)
        return dictionary, reverse_dictionary

    def decompose_paths(self, paths, dictionary):
        """
        Decompose the paths to variable length sub-paths keeping the first tool fixed
        """
        max_len = 0
        sub_paths_pos = list()
        for index, item in enumerate(paths):
            tools = item.split(",")
            len_tools = len(tools)
            if len_tools > max_len:
                max_len = len_tools
            if len_tools < self.max_tool_sequence_len:
                #for window in range(1, len_tools):
                #sequence = tools[0: window + 1]
                sequence = tools[0: len_tools]
                tools_pos = [str(dictionary[str(tool_item)]) for tool_item in sequence]
                if len(tools_pos) > 1:
                    sub_paths_pos.append(",".join(tools_pos))
        #print(len(sub_paths_pos))
        sub_paths_pos = list(set(sub_paths_pos))
        print("Max length of tools: ", max_len)
        #print(len(sub_paths_pos))
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


    def prepare_input_target_paths(self, dictionary, reverse_dictionary, paths):
        input_target_paths = dict()
        for item in paths:
            input_tools = item.split(",")
            target_tools = ",".join(input_tools[1:])
            input_target_paths[item] = target_tools
        return input_target_paths


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
            print(train_seq, train_label)
            start_pos = self.max_tool_sequence_len - len(positions)
            for id_pos, pos in enumerate(positions):
                data_mat[train_counter][start_pos + id_pos] = int(pos)
                data_mat[train_counter][id_pos] = int(pos)
            for label_item in train_label.split(","):
                label_mat[train_counter][int(label_item)] = 1.0
            train_counter += 1
        return data_mat, label_mat


    def pad_paths(self, paths_dictionary, num_classes, standard_connections, reverse_dictionary, dictionary):
        """
        Add padding to the tools sequences and create multi-hot encoded labels
        """
        size_data = len(paths_dictionary)
        input_mat = np.zeros([size_data, self.max_tool_sequence_len])
        target_mat = np.zeros([size_data, self.max_tool_sequence_len])
        print(input_mat.shape)
        print(target_mat.shape)
        train_counter = 0
        for input_seq, target_seq in list(paths_dictionary.items()):
            input_seq_tools = input_seq.split(",")
            target_seq_tools = target_seq.split(",")
            input_seq_tools.insert(0, dictionary[start_token_name])
            target_seq_tools.insert(0, dictionary[start_token_name])
            #start_pos = self.max_tool_sequence_len - len(input_seq_tools)
            #input_mat[train_counter][0] = index_start_token
            #target_mat[train_counter][0] = index_start_token
            #print(input_seq_tools, target_seq_tools)
            for id_pos, pos in enumerate(input_seq_tools):
                #input_mat[train_counter][start_pos + id_pos] = int(pos)
                input_mat[train_counter][id_pos] = int(pos)
            for id_pos, pos in enumerate(target_seq_tools):
                #target_mat[train_counter][start_pos + id_pos + 1] = int(pos)
                target_mat[train_counter][id_pos] = int(pos)
            #print(input_mat[train_counter])
            #print(target_mat[train_counter])
            #print()
            train_counter += 1
        return input_mat, target_mat


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


    def get_train_last_tool_freq(self, train_paths, reverse_dictionary):
        """
        Get the frequency of last tool of each tool sequence
        to estimate the frequency of tool sequences
        """
        last_tool_freq = dict()
        freq_dict_names = dict()
        for path in train_paths:
            last_tool = path.split(",")[-1]
            if last_tool not in last_tool_freq:
                last_tool_freq[last_tool] = 0
                freq_dict_names[reverse_dictionary[int(last_tool)]] = 0
            last_tool_freq[last_tool] += 1
            freq_dict_names[reverse_dictionary[int(last_tool)]] += 1
        utils.write_file("data/freq_dict_names.txt", freq_dict_names)
        return last_tool_freq


    def get_toolid_samples(self, train_data, l_tool_freq):
        l_tool_tr_samples = dict()
        for tool_id in l_tool_freq:
            for index, tr_sample in enumerate(train_data):
                last_tool_id = str(int(tr_sample[-1]))
                if last_tool_id == tool_id:
                    if last_tool_id not in l_tool_tr_samples:
                        l_tool_tr_samples[last_tool_id] = list()
                    l_tool_tr_samples[last_tool_id].append(index)
        return l_tool_tr_samples


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

        all_paths = ",".join(all_unique_paths)
        utils.write_file("data/all_paths.txt", all_paths)

        print("Creating dictionaries...")
        #multilabels_paths = self.prepare_paths_labels_dictionary(dictionary, rev_dict, all_unique_paths, compatible_next_tools)
        multilabels_paths = self.prepare_input_target_paths(dictionary, rev_dict, all_unique_paths)

        print("Complete data: %d" % len(multilabels_paths))
        train_paths_dict, test_paths_dict = self.split_test_train_data(multilabels_paths)

        utils.write_file("data/rev_dict.txt", rev_dict)
        utils.write_file("data/f_dict.txt", dictionary)
        utils.write_file("data/test_paths_dict.txt", test_paths_dict)

        print("Train data: %d" % len(train_paths_dict))
        print("Test data: %d" % len(test_paths_dict))

        print("Padding train and test data...")
        # pad training and test data with leading zeros
        test_data, test_labels = self.pad_paths(test_paths_dict, num_classes, standard_connections, rev_dict, dictionary)
        train_data, train_labels = self.pad_paths(train_paths_dict, num_classes, standard_connections, rev_dict, dictionary)

        print(train_data[0:5])
        print()
        print(train_labels[0:5])

        train_data = train_data[:20000]
        train_labels = train_labels[:20000]

        test_data = test_data[:2000]
        test_labels = test_labels[:2000]

        print("Train data: ", train_data.shape)
        print("Test data: ", test_data.shape)
        
        '''print("Estimating sample frequency...")
        l_tool_freq = self.get_train_last_tool_freq(train_paths_dict, rev_dict)
        l_tool_tr_samples = self.get_toolid_samples(train_data, l_tool_freq)

        # Predict tools usage
        print("Predicting tools' usage...")
        usage_pred = predict_tool_usage.ToolPopularity()
        usage = usage_pred.extract_tool_usage(tool_usage_path, cutoff_date, dictionary)
        tool_usage_prediction = usage_pred.get_pupularity_prediction(usage)
        t_pred_usage = self.get_predicted_usage(dictionary, tool_usage_prediction)
        # get class weights using the predicted usage for each tool
        class_weights = self.assign_class_weights(num_classes, t_pred_usage)'''

        #return train_data, train_labels, test_data, test_labels, dictionary, rev_dict, class_weights, t_pred_usage, l_tool_freq, l_tool_tr_samples
        return train_data, train_labels, test_data, test_labels, dictionary, rev_dict
