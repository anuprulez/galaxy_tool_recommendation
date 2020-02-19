#!/bin/bash

python scripts/main.py -wf data/workflow-connections-subset.tsv -tu data/tool-popularity-19-09.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 20 -oe 5 -me 5 -ts 0.2 -vs 0.2 -bs '1,512' -ut '1,512' -es '1,512' -dt '0.0,0.5' -sd '0.0,0.5' -lr '0.00001,0.1' -ad 'elu' -ao 'sigmoid' -cpus 8

