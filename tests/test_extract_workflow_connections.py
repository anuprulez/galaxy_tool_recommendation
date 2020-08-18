import unittest

import os
import sys
sys.path.append("../scripts/")

from scripts import utils
from scripts import extract_workflow_connections

class TestWorkflows(unittest.TestCase):
    """ """

    def test_all(self):
        print("Test passed")


if __name__ == "__main__":
    unittest.main()
