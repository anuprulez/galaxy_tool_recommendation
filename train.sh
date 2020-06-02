#!/bin/bash

python scripts/main.py -wf data/worflow-connection-subset-20-04.tsv -tu data/tool-popularity-20-04.tsv -om data/tool_recommendation_model_no_reg.hdf5 -cd '2017-12-01' -pl 25 -ep 10 -oe 5 -me 20 -ts 0.2 -bs '32,256' -ut '32,512' -es '32,512' -lr '0.00001,0.1' -cpus 4

