# Tool recommender system in Galaxy using deep learning (Gated recurrent units neural network)

[![DOI](https://zenodo.org/badge/241409061.svg)](https://zenodo.org/badge/latestdoi/241409061)

## General information

Project name: Galaxy tool recommendation

Project home page: https://github.com/anuprulez/galaxy_tool_recommendation

Data: https://github.com/anuprulez/galaxy_tool_recommendation/tree/master/data

Operating system(s): Linux

Programming language: Python

Scripts: https://github.com/anuprulez/galaxy_tool_recommendation/tree/master/scripts

iPython notebook: https://github.com/anuprulez/galaxy_tool_recommendation/blob/master/ipython_script/tool_recommendation_gru_wc.ipynb

Other requirements: python=3.6, tensorflow=1.13.1, keras=2.3.0, scikit-learn=0.21.3, numpy=1.17.2, h5py=2.9.0, csvkit=1.0.4, hyperopt=0.1.2, matplotlib=3.1.1

Training script: https://github.com/anuprulez/galaxy_tool_recommendation/blob/master/train.sh

License: MIT License

RRID: SCR_018491

bioToolsID: tool_recommender_system_in_galaxy

## History of the work

Initial work to create tool recommendation model is stored at https://github.com/anuprulez/similar_galaxy_workflow. This repository storing the history of work until October, 2019 will not be used in future. The current repository (https://github.com/anuprulez/galaxy_tool_recommendation) will be used for current and future developments.

## Description of other branches in this repository
- Tool recommender system in Galaxy using deep learning (Gated recurrent units neural network) (branch: [master](https://github.com/anuprulez/galaxy_tool_recommendation))
- Tool recommender system in Galaxy using deep learning (Dense neural network) (branch: [dnn_wc](https://github.com/anuprulez/galaxy_tool_recommendation/tree/dnn_wc))
- Tool recommender system in Galaxy using deep learning (Convolutional neural network) (branch: [cnn_wc](https://github.com/anuprulez/galaxy_tool_recommendation/tree/cnn_wc))
- Tool Recommender in Galaxy using stored tool sequences (branch: [statistical_model](https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model))
- Tool recommender system in Galaxy using ExtraTrees classifier (branch: [sklearn_rf](https://github.com/anuprulez/galaxy_tool_recommendation/tree/sklearn_rf))
- Tool recommender system in Galaxy using deep learning (Gated recurrent units neural network with no regularisation) (branch: [no_regularisation](https://github.com/anuprulez/galaxy_tool_recommendation/tree/no_regularisation))

## (To reproduce this work) How to create a sample tool recommendation model:

**Note**: To reproduce this work after training on complete model, it is required to have a decent compute resource (with at least 10 GB RAM) and it takes > 24 hrs to create a trained model on complete set of workflows (~ 18,000). However, the following steps can be used to create a sample tool recommendation model on a subset of workflows:

1. Install the dependencies by executing the following lines:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_gru_wc`

2. Execute `sh train.sh` (https://github.com/anuprulez/galaxy_tool_recommendation/blob/master/train.sh). It runs on a subset of workflows. Use file `data/worflow-connection-20-04.tsv` in the training script to train on complete set of workflows (It takes a long time to finish).

3. After successful finish (~2-3 minutes), a trained model is created at `data/<<file name>>.hdf5`.

4. Put this trained model file at `ipython_script/data/<<file name>>.hdf5` and execute the ipython notebook.

5. A model trained on all workflows is present at `ipython_script/data/tool_recommendation_model_20_05.hdf5` which can be used to predict tools using the IPython notebook `ipython_script/tool_recommendation_gru_wc.ipynb`

## Data description:

Execute data extraction script `extract_data.sh` to extract two tabular files - `tool-popularity-20-04.tsv` and `worflow-connection-20-04.tsv`. This script should be executed on a Galaxy instance's database (ideally should be executed by a Galaxy admin). There are two methods in the script one each to generate two tabular files. The first file (`tool-popularity-20-04.tsv`) contains information about the usage of tools per month. The second file (`worflow-connection-20-04.tsv`) contains workflows present as the connections of tools. Save these tabular files. These tabular files are present under `/data` folder and can be used to run deep learning training by following steps.

### Description of all parameters mentioned in the training script:

`python <main python script> -wf <path to workflow file> -tu <path to tool usage file> -om <path to the final model file> -cd <cutoff date> -pl <maximum length of tool path> -ep <number of training iterations> -oe <number of iterations to optimise hyperparamters> -me <maximum number of evaluation to optimise hyperparameters> -ts <fraction of test data> -bs <range of batch sizes> -ut <range of hidden units> -es <range of embedding sizes> -dt <range of dropout> -sd <range of spatial dropout> -rd <range of recurrent dropout> -lr <range of learning rates> -cpus <number of CPUs>`

   - `<main python script>`: This script is the entry point of the entire analysis. It is present at `scripts/main.py`.
   
   - `<path to workflow file>`: It is a path to a tabular file containing Galaxy workflows. E.g. `data/worflow-connection-20-04.tsv`.
   
   - `<path to tool popularity file>`: It is a path to a tabular file containing usage frequencies of Galaxy tools. E.g. `data/tool-popularity-20-04.tsv`.
   
   - `<path to trained model file>`: It is a path of the final trained model (`h5` file). E.g. `data/<<file name>>.hdf5`.
   
   - `<cutoff date>`: It is used to set the earliest date from which the usage frequencies of tools should be considered. The format of the date is YYYY-MM-DD. This date should be in the past. E.g. `2017-12-01`.

   - `<maximum length of tool path>`: This takes an integer and specifies the maximum size of a tool sequence extracted from any workflow. Any tool sequence of length larger than this number is not included in the dataset for training. E.g. `25`.

   - `<number of training iterations>`: Once the best configuration of hyperparameters has been found, the neural network takes this configuration and runs for "n_epochs" number of times minimising the error to produce a model at the end. E.g. `10`.

   - `<number of iterations to optimise hyperparamters>`: This number specifies how many iterations would the neural network executes to evaluate each sampled configuration. E.g. `5`.

   - `<maximum number of evaluation to optimise hyperparameters>`: The hyperparameters of the neural network are tuned using a Bayesian optimisation approach and multiple configurations are sampled from different ranges of parameters. The number specified in this parameter is the number of configurations of hyperparameters evaluated to optimise them. Higher the number, the longer is the running time of the tool. E.g. `20`.

   - `<fraction of test data>`: It specifies the size of the test set. For example, if it is 0.5, then the test set is half of the entire data available. It should not be set to more than 0.5. This set is used for evaluating the precision on an unseen set. E.g. `0.2`.

   - `<range of batch sizes>`:  The training of the neural network is done using batch learning in this work. The training data is divided into equal batches and for each epoch (a training iteration), all batches of data are trained one after another. A higher or lower value can unsettle the training. Therefore, this parameter should be optimised. E.g. `32,256`.

   - `<range of hidden units>`: This number is the number of hidden recurrent units. A higher number means stronger learning (may lead to overfitting) and a lower number means weaker learning (may lead to underfitting). Therefore, this number should be optimised. E.g. `32,512`.

   - `<range of embedding sizes>`: For each tool, a fixed-size vector is learned and this fixed-size is known as the embedding size. This size remains same for all the tools. A lower number may underfit and a higher number may overfit. This parameter should be optimised as well. E.g. `32,512`.

   - `<range of dropout>`: A neural network tends to overfit (especially when it is stronger). Therefore, to avoid or minimize overfitting, dropout is used. The fraction specified by dropout is the fraction of units "deleted" randomly from the network to impose randomness which helps in avoiding overfitting. This parameter should be optimised as well. E.g. `0.0,0.5`.
    
   - `<range of spatial dropout>`: Similar to dropout, this is used to reduce overfitting in the embedding layer. This parameter should be optimised as well. E.g. `0.0,0.5`.

   - `<range of recurrent dropout>`: Similar to dropout and spatial dropout, this is used to reduce overfitting in the recurrent layers (hidden). This parameter should be optimised as well. E.g. `0.0,0.5`.

   - `<range of learning rates>`: The learning rate specifies the speed of learning. A higher value ensures fast learning (the optimiser may diverge) and a lower value causes slow learning (may not reach the optimum). This parameter should be optimised as well. E.g. `0.0001, 0.1`.

   - `<number of CPUs>`: This takes the number of CPUs to be allocated to parallelise the training of the neural network. E.g. `4`.

### (To reproduce this work on complete set of workflows) Example command:

   `python scripts/main.py -wf data/worflow-connection-20-04.tsv -tu data/tool-popularity-20-04.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 10 -oe 5 -me 20 -ts 0.2 -bs '32,256' -ut '32,256' -es '32,256' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001,0.1' -cpus 4`

Once the script finishes, `H5` model file is created at the given location (`path to trained model file`).

## (For Galaxy admins) The following steps are only necessary for deploying on any Galaxy server:

1. (Already done!) The latest model is uploaded at: https://github.com/galaxyproject/galaxy-test-data/blob/master/tool_recommendation_model.hdf5. Change this path only if there is a different model.

2. In the `galaxy.yml.sample` config file, make the following changes:
    - Enable and then set the property `enable_tool_recommendations` to `true`.

3. In order to allow Galaxy admins to add/remove tools from the list of recommendations, the following steps can be used:
    - A Galaxy config file has been provided (https://github.com/galaxyproject/galaxy/blob/dev/config/tool_recommendations_overwrite.yml.sample) to offer following features and instructions to use these features are given in the file itself:
        - Enable `admin_tool_recommendations_path` in Galaxy's config file at `config/galaxy.yml.sample`.
        - Add tool(s) and mark them "deprecated".
        - Add new tool(s) to the list of recommendations.
        - Overwrite all recommendations (predicted by trained model). (Enable `overwrite_model_recommendations` and set to `true` in Galaxy's config file at `config/galaxy.yml.sample`).

## For Galaxy end-users:

Open the workflow editor and choose any tool from the toolbox. Then, hover on the `right-arrow` icon in top-right of the tool to see the recommended tools in a pop-over. Moreover, execute a tool and see recommended tools for further analysis in a tree visualisation.

## For contributors:

Information about contributors and how to contribute is present in `CONTRIBUTING.md` file.
