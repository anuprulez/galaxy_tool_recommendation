#!/bin/bash

python scripts/main.py -wf data/worflow-connection-04-20-10000.tsv -tu data/tool-popularity-19-09.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 1 -oe 1 -me 1 -ts 0.2 -vs 0.2 -bs '128' -ut '64,256' -es '64,256' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001,0.1' -ar 'elu' -ao 'sigmoid' -cpus 4
