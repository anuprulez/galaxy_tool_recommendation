#!/bin/bash

python scripts/main.py -wf data/workflow-connections-subset.tsv -tu data/tool-popularity-19-09.tsv -om data/tool_recommendation_attention_model.hdf5 -cd '2017-12-01' -pl 25 -ep 5 -oe 5 -me 10 -ts 0.2 -vs 0.2 -bs '32,256' -ut '32,256' -es '32,256' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001,0.1' -cp 4
