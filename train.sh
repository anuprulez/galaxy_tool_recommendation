#!/bin/bash

python scripts/main.py -wf data/aug_22/workflow-connections_Aug_22.csv -tu data/aug_22/tool_popularity_Aug_22.csv -om data/feb_22/tool_recommendation_model.hdf5 -cd '2021-12-31' -pl 25 -ep 5 -oe 5 -me 2 -ts 0.2 -bs '4,8' -ut '32,35' -es '32,35' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001,0.1' -ud false -cpus 4
