import unittest

from scripts import extract_workflow_connections
from scripts import prepare_data


class TestRecommendationSystemData(unittest.TestCase):

    def test_recommendation_system_data(self):
        workflow_file_path = "scripts/tests/test_data/worflow-connection-subset-20-04_test.tsv"
        tool_usage_file_path = "scripts/tests/test_data/tool-popularity-20-04_test.tsv"
        maximum_path_length = 25
        test_share = 0.1
        cutoff_date = "2017-12-01"

        # test conversion of workflows into paths
        extract_wf = extract_workflow_connections.ExtractWorkflowConnections()
        unique_paths, compatible_next_tools, standard_connections = extract_wf.read_tabular_file(workflow_file_path)
        assert len(unique_paths) > 0
        assert len(compatible_next_tools) > 0
        assert len(standard_connections) > 0

        # test conversion of paths into matrices and dictionaries
        data = prepare_data.PrepareData(maximum_path_length, test_share)
        train_data, train_labels, test_data, test_labels, data_dictionary, reverse_dictionary, class_weights, usage_pred, train_tool_freq, tool_tr_samples = data.get_data_labels_matrices(unique_paths, tool_usage_file_path, cutoff_date, compatible_next_tools, standard_connections)

        assert len(train_data) > 0
        assert len(train_labels) > 0
        assert len(test_data) > 0
        assert len(test_labels) > 0
        assert len(data_dictionary) > 0
        assert len(reverse_dictionary) > 0
        assert len(class_weights) > 0
        assert len(usage_pred) > 0
        assert len(train_tool_freq) > 0
        assert len(tool_tr_samples) > 0


if __name__ == "__main__":
    unittest.main()
