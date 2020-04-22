#!/bin/bash

python scripts/main.py -wf data/workflow-connections-19-09.tsv -tu data/tool-popularity-19-09.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25
