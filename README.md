# Tool recommender system in Galaxy using deep learning (Gated recurrent units neural network)

## General information

Project name: Galaxy tool recommendation

Project home page: https://github.com/anuprulez/galaxy_tool_recommendation

Data: https://github.com/anuprulez/galaxy_tool_recommendation/tree/master/data

Operating system(s): Linux

Programming language: Python

Scripts: https://github.com/anuprulez/galaxy_tool_recommendation/tree/master/scripts

Other requirements: python=3.6, tensorflow=1.13.1, keras=2.3.0, scikit-learn=0.21.3, numpy=1.17.2, h5py=2.9.0, csvkit=1.0.4, hyperopt=0.1.2,matplotlib=3.1.1

License: MIT License


## How to execute the script

1. Execute the script `extract_data.sh` to extract two tabular files - `tool-popularity.tsv` and `wf-connections.tsv`. This script should be executed on a Galaxy instance's database (ideally should be executed by a Galaxy admin). There are two methods in the script one each to generate two tabular files. The first file (`tool-popularity.tsv`) contains information about the usage of tools per month. The second file (`wf-connections.tsv`) contains workflows present as the connections of tools. Save these tabular files. These tabular files are present under `/data` folder and can be used to run deep learning training by following steps.

2. Install the dependencies by executing the following lines:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction`

3. Execute the file `train.sh`.

Note: if executed on complete dataset `wf-connections.tsv`, it will take a long time (> 24 hrs) to finish and needs a large memory. A subset of this file is provided to run the training at `data/workflow-connections-subset.tsv`. Please replace the original workflow file with this smaller one. Once hyperparameter optimisation and actual training finish, a trained model is created at the path specified in `train.sh` script. Please know that this would be a scaled-down model. Trained model (`tool_recommendation_model.hdf5`) with complete data is present at `ipython_script/data/` which can be used to predict tool using the IPython notebook `ipython_script/tool_recommendation_gru_wc.ipynb`).

The training script has following input parameters:

    `python <main python script> -wf <path to workflow file> -tu <path to tool usage file> -om <path to the final model file> -cd <cutoff date> -pl <maximum length of tool path> -ep <number of training iterations> -oe <number of iterations to optimise hyperparamters> -me <maximum number of evaluation to optimise hyperparameters> -ts <fraction of test data> -vs <fraction of validation data> -bs <range of batch sizes> -ut <range of hidden units> -es <range of embedding sizes> -dt <range of dropout> -sd <range of spatial dropout> -rd <range of recurrent dropout> -lr <range of learning rates> -ar <name of recurrent activation> -ao <name of output activation> -cpus <number of CPUs>`
    
### Description of all parameters mentioned in the training script:

   - `<main python script>`: This script is the entry point of the entire analysis. It is present at `scripts/main.py`.
   - `<path to workflow file>`: It is a path to a tabular file containing Galaxy workflows. E.g. `data/wf-connections.tsv`.
   - `<path to tool popularity file>`: It is a path to a tabular file containing usage frequencies of Galaxy tools. E.g. `data/tool-popularity.tsv`.
   - `<path to trained model file>`: It is a path of the final trained model (`h5` file). E.g. `data/trained_model.hdf5`.
    
   - `<cutoff date>`: It is used to set the earliest date from which the usage frequencies of tools should be considered. The format of the date is YYYY-MM-DD. This date should be in the past. E.g. `2017-12-01`.
    
   - `<maximum length of tool path>`: This takes an integer and specifies the maximum size of a tool sequence extracted from any workflow. Any tool sequence of length larger than this number is not included in the dataset for training. E.g. `25`.
    
   - `<maximum number of evaluation to optimise hyperparameters>`: The hyperparameters of the neural network are tuned using a Bayesian optimisation approach and multiple configurations are sampled from different ranges of parameters. The number specified in this parameter is the number of configurations of hyperparameters evaluated to optimise them. Higher the number, the longer is the running time of the tool. E.g. `30`.
    
   - `<number of iterations to optimise hyperparamters>`: This number specifies how many iterations would the neural network executes to evaluate each sampled configuration. E.g. `10`.
    
   - `<number of training iterations>`: Once the best configuration of hyperparameters has been found, the neural network takes this configuration and runs for "n_epochs" number of times minimising the error to produce a model at the end. E.g. `10`.
    
   - `<fraction of test data>`: It specifies the size of the test set. For example, if it is 0.5, then the test set is half of the entire data available. It should not be set to more than 0.5. This set is used for evaluating the precision on an unseen set. E.g. `0.2`.
    
   - `<fraction of validation data>`: It specifies the size of the validation set. For example, if it is 0.5, then the validation set is half of the entire data available. It should not be set to more than 0.5. This set is used for computing error while training on the best configuration. It should always be greater than 0.0. E.g. `0.2`.
    
   - `<range of batch sizes>`:  The training of the neural network is done using batch learning in this work. The training data is divided into equal batches and for each epoch (a training iteration), all batches of data are trained one after another. A higher or lower value can unsettle the training. Therefore, this parameter should be optimised. E.g. `1,128`.
    
   - `<range of hidden units>`: This number is the number of hidden recurrent units. A higher number means stronger learning (may lead to overfitting) and a lower number means weaker learning (may lead to underfitting). Therefore, this number should be optimised. E.g. `1,128`.
    
   - `<range of embedding sizes>`: For each tool, a fixed-size vector is learned and this fixed-size is known as the embedding size. This size remains same for all the tools. A lower number may underfit and a higher number may overfit. This parameter should be optimised as well. E.g. `1,128`.
    
   - `<range of dropout>`: A neural network tends to overfit (especially when it is stronger). Therefore, to avoid or minimize overfitting, dropout is used. The fraction specified by dropout is the fraction of units "deleted" randomly from the network to impose randomness which helps in avoiding overfitting. This parameter should be optimised as well. E.g. `0.0,0.5`.
    
   - `<range of spatial dropout>`: Similar to dropout, this is used to reduce overfitting in the embedding layer. This parameter should be optimised as well. E.g. `0.0,0.5`.
    
   - `<range of recurrent dropout>`: Similar to dropout and spatial dropout, this is used to reduce overfitting in the recurrent layers (hidden). This parameter should be optimised as well. E.g. `0.0,0.5`.
    
   - `<range of learning rates>`: The learning rate specifies the speed of learning. A higher value ensures fast learning (the optimiser may diverge) and a lower value causes slow learning (may not reach the optimum). This parameter should be optimised as well. E.g. `0.0001, 0.1`.
    
   - `<name of recurrent activation>`: Activations are mathematical functions to transform input into output. This takes the name of an activation function from the list of Keras activations (https://keras.io/activations/) for recurrent layers. E.g. `elu`.
    
   - `<name of output activation>`: This takes the activation for transforming the input of the last layer to the output of the neural network. It is also taken from Keras activations (https://keras.io/activations/). E.g. `sigmoid`.
    
   - `<number of CPUs>`: This takes the number of CPUs to be allocated to parallelise the training of the neural network. E.g. `4`.

    An example command: `python scripts/main.py -wf data/workflow-connections-19-09.tsv -tu data/tool-popularity-19-09.tsv -om data/trained_model_sample.hdf5 -cd '2017-12-01' -pl 25 -ep 20 -oe 20 -me 50 -ts 0.0 -vs 0.2 -bs '1,512' -ut '1,512' -es '1,512' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001,0.1' -ar 'elu' -ao 'sigmoid' -cpus 4`

4. The training of the neural network takes a long time (> 24 hours) for the complete data. Once the script finishes, `h5` model file is created at the given location (`path to trained model file`).

## The following steps are only necessary for deploying on Galaxy server.

5. Place the new model in the Galaxy repository at `galaxy/database/trained_model.hdf5`. 

6. In the `galaxy.yml` config file, make the following changes:
    - Enable and then set the property `enable_tool_recommendations` to `true`
    - Enable and then set the property `tool_recommendation_model_path` to `database/trained_model.hdf5`

7. Now go to the workflow editor and choose any tool from the toolbox. Then, you can see a `right-arrow` in top-right of the tool. Click on it to see the recommended tools to be used after the previously chosen tool.
