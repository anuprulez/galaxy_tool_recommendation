# Tool recommender system in Galaxy using deep learning (Dense neural network)

## General information

Project name: Galaxy tool recommendation

Project home page: https://github.com/anuprulez/galaxy_tool_recommendation/tree/dnn_wc

Data: https://github.com/anuprulez/galaxy_tool_recommendation/tree/dnn_wc/data

Operating system(s): Linux

Programming language: Python

Scripts: https://github.com/anuprulez/galaxy_tool_recommendation/tree/dnn_wc/scripts

iPython notebook: https://github.com/anuprulez/galaxy_tool_recommendation/blob/dnn_wc/ipython_script/tool_recommendation_dnn_wc.ipynb

Other requirements: python=3.6, tensorflow=1.13.1, keras=2.3.0, scikit-learn=0.21.3, numpy=1.17.2, h5py=2.9.0, csvkit=1.0.4, hyperopt=0.1.2,matplotlib=3.1.1

License: MIT License

## How to create a sample tool recommendation model

As the deep learning training time is high (> 12 hrs), the following steps should be used to create a sample tool recommendation model on a subset of workflows:

1. Install the dependencies by executing the following lines:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_dnn_wc`
    
2. Execute `sh train.sh` (https://github.com/anuprulez/galaxy_tool_recommendation/blob/dnn_wc/train.sh).

3. After successful finish (~2-3 minutes), a trained model is created at `data/tool_recommendation_model.hdf5`.

4. A model trained on all workflows is present at `ipython_script/data/` which can be used to predict tools using the IPython notebook 
`ipython_script/tool_recommendation_dnn_wc.ipynb`

## How to run the project on complete data to create tool recommendation model

1. Execute data extraction script `extract_data.sh` to extract two tabular files - `tool-popularity-19-09.tsv` and `workflow-connections-19-09.tsv`. This script should be executed on a Galaxy instance's database (ideally should be executed by a Galaxy admin). There are two methods in the script one each to generate two tabular files. The first file (`tool-popularity-19-09.tsv`) contains information about the usage of tools per month. The second file (`workflow-connections-19-09.tsv`) contains workflows present as the connections of tools. Save these tabular files. These tabular files are present under `/data` folder and can be used to run deep learning training by following steps.

2. Install the dependencies by executing the following lines if not done before:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_dnn_wc`

3. Execute script `train.sh`. Please check that the complete workflow file (`workflow-connections-19-09.tsv`) is being used in the training script.

The training script has following input parameters:

    `python <main python script> -wf <path to workflow file> -tu <path to tool usage file> -om <path to the final model file> -cd <cutoff date> -pl <maximum length of tool path> -ep <number of training iterations> -oe <number of iterations to optimise hyperparamters> -me <maximum number of evaluation to optimise hyperparameters> -ts <fraction of test data> -vs <fraction of validation data> -bs <range of batch sizes> -ut <range of hidden units> -es <range of embedding sizes> -dt <range of dropout> -sd <range of spatial dropout> -lr <range of learning rates> -ad <name of dense activation> -ao <name of output activation> -cpus <number of CPUs>`
    
### Description of all parameters mentioned in the training script:

   - `<main python script>`: This script is the entry point of the entire analysis. It is present at `scripts/main.py`.
   - `<path to workflow file>`: It is a path to a tabular file containing Galaxy workflows. E.g. `data/workflow-connections-19-09.tsv`.
   - `<path to tool popularity file>`: It is a path to a tabular file containing usage frequencies of Galaxy tools. E.g. `data/tool-popularity-19-09.tsv`.
   - `<path to trained model file>`: It is a path of the final trained model (`h5` file). E.g. `data/tool_recommendation_model.hdf5`.
    
   - `<cutoff date>`: It is used to set the earliest date from which the usage frequencies of tools should be considered. The format of the date is YYYY-MM-DD. This date should be in the past. E.g. `2017-12-01`.
    
   - `<maximum length of tool path>`: This takes an integer and specifies the maximum size of a tool sequence extracted from any workflow. Any tool sequence of length larger than this number is not included in the dataset for training. E.g. `25`.
    
   - `<maximum number of evaluation to optimise hyperparameters>`: The hyperparameters of the neural network are tuned using a Bayesian optimisation approach and multiple configurations are sampled from different ranges of parameters. The number specified in this parameter is the number of configurations of hyperparameters evaluated to optimise them. Higher the number, the longer is the running time of the tool. E.g. `30`.
    
   - `<number of iterations to optimise hyperparamters>`: This number specifies how many iterations would the neural network executes to evaluate each sampled configuration. E.g. `10`.
    
   - `<number of training iterations>`: Once the best configuration of hyperparameters has been found, the neural network takes this configuration and runs for "n_epochs" number of times minimising the error to produce a model at the end. E.g. `10`.
    
   - `<fraction of test data>`: It specifies the size of the test set. For example, if it is 0.5, then the test set is half of the entire data available. It should not be set to more than 0.5. This set is used for evaluating the precision on an unseen set. E.g. `0.2`.
    
   - `<fraction of validation data>`: It specifies the size of the validation set. For example, if it is 0.5, then the validation set is half of the entire data available. It should not be set to more than 0.5. This set is used for computing error while training on the best configuration. It should always be greater than 0.0. E.g. `0.2`.
    
   - `<range of batch sizes>`:  The training of the neural network is done using batch learning in this work. The training data is divided into equal batches and for each epoch (a training iteration), all batches of data are trained one after another. A higher or lower value can unsettle the training. Therefore, this parameter should be optimised. E.g. `1,128`.
    
   - `<range of hidden units>`: This number is the number of hidden units for dense layers. A higher number means stronger learning (may lead to overfitting) and a lower number means weaker learning (may lead to underfitting). Therefore, this number should be optimised. E.g. `1,128`.
    
   - `<range of embedding sizes>`: For each tool, a fixed-size vector is learned and this fixed-size is known as the embedding size. This size remains same for all the tools. A lower number may underfit and a higher number may overfit. This parameter should be optimised as well. E.g. `1,128`.
    
   - `<range of dropout>`: A neural network tends to overfit (especially when it is stronger). Therefore, to avoid or minimize overfitting, dropout is used. The fraction specified by dropout is the fraction of units "deleted" randomly from the network to impose randomness which helps in avoiding overfitting. This parameter should be optimised as well. E.g. `0.0,0.5`.
    
   - `<range of spatial dropout>`: Similar to dropout, this is used to reduce overfitting in the embedding layer. This parameter should be optimised as well. E.g. `0.0,0.5`.
    
   - `<range of learning rates>`: The learning rate specifies the speed of learning. A higher value ensures fast learning (the optimiser may diverge) and a lower value causes slow learning (may not reach the optimum). This parameter should be optimised as well. E.g. `0.0001, 0.1`.
    
   - `<name of dense activation>`: Activations are mathematical functions to transform input into output. This takes the name of an activation function from the list of Keras activations (https://keras.io/activations/) for dense layers. E.g. `elu`.
    
   - `<name of output activation>`: This takes the activation for transforming the input of the last layer to the output of the neural network. It is also taken from Keras activations (https://keras.io/activations/). E.g. `sigmoid`.
    
   - `<number of CPUs>`: This takes the number of CPUs to be allocated to parallelise the training of the neural network. E.g. `4`.

  An example command: `python scripts/main.py -wf data/workflow-connections-19-09.tsv -tu data/tool-popularity-19-09.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 20 -oe 5 -me 5 -ts 0.2 -vs 0.2 -bs '1,512' -ut '1,512' -es '1,512' -dt '0.0,0.5' -sd '0.0,0.5' -lr '0.00001,0.1' -ad 'elu' -ao 'sigmoid' -cpus 8`

4. The training of the neural network takes a long time (> 12 hours) for the complete data. Once the script finishes, `h5` model file is created at the given location (`path to trained model file`).

## The following steps are only necessary for deploying on Galaxy server.

5. Upload new model to: https://github.com/anuprulez/download_store/tree/tool_recommendation_model/tool_recommendation_model. 

6. In the `galaxy.yml.sample` config file, make the following changes:
    - Enable and then set the property `enable_tool_recommendations` to `true`.
    - Enable and then set the property `tool_recommendation_model_path` to `https://github.com/anuprulez/download_store/tree/tool_recommendation_model/tool_recommendation_model`.

7. Now go to the workflow editor and choose any tool from the toolbox. Then, you can see a `right-arrow` in top-right of the tool. Click on it to see the recommended tools to be used after the previously chosen tool.
