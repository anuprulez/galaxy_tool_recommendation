#!/bin/bash

python scripts/main.py -wf data/worflow-connection-subset-20-04.tsv -tu data/tool-popularity-20-04.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -me 5 -ts 0.2 -ne '1,50' -ct 'gini,entropy' -md '1,10' -mss '0.0001,1.0' -mf 'auto,sqrt,log2' -cpus 4
