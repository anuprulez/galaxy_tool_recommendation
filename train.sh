#!/bin/bash

python scripts/main.py -wf data/feb_22/wf_frame_subset_July_12_all.csv -tu data/feb_22/tool_popularity_march_22.csv -om data/feb_22/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 5 -oe 5 -me 2 -ts 0.2 -bs '4,8' -ut '32,35' -es '32,35' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001,0.1' -cpus 4
