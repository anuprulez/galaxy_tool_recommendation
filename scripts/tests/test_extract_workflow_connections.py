import unittest

from scripts import utils
from scripts import extract_workflow_connections

class TestExtractWorkflowConnections(unittest.TestCase):

    def test_read_tabular_file(self):
        workflow_file_path = "scripts/tests/test_data/worflow-connection-subset-20-04_test.tsv"
        extract_wf = extract_workflow_connections.ExtractWorkflowConnections()
        unique_paths, compatible_next_tools, standard_connections = extract_wf.read_tabular_file(workflow_file_path)
        assert len(unique_paths) > 0
        assert len(compatible_next_tools) > 0
        assert len(standard_connections) == 0


if __name__ == "__main__":
    unittest.main()
