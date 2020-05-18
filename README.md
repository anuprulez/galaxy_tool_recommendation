# Tool Recommender in Galaxy using stored tool sequences

## General information

Project name: Galaxy tool recommendation

Project home page: https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model

Data: https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model/data

Operating system(s): Linux

Programming language: Python

Scripts: https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model/scripts

iPython notebook: https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model/ipython_script/tool_recommendation_statistical_model.ipynb

Other requirements: python=3.6, scikit-learn=0.21.3, h5py=2.9.0, csvkit=1.0.4, nb_conda

Training script: https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model/train.sh

License: MIT License

## How to create a sample tool recommendation model

Following steps should be used to create a sample tool recommendation model on complete set of workflows:

1. Install the dependencies by executing the following lines:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_statistical_model`

2. Execute `sh train.sh` (https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model/train.sh).

3. After successful finish (a few minutes), a model is created at `data/tool_recommendation_model_statistical_model.hdf5`.

4. A model created using all workflows is present at `ipython_script/data/` which can be used to predict tools using the IPython notebook 
`ipython_script/tool_recommendation_statistical_model.ipynb`

## How to run the project on complete data to create tool recommendation model

1. Execute data extraction script `extract_data.sh` to extract two tabular files - `tool-popularity-20-04.tsv` and `worflow-connection-20-04.tsv`. This script should be executed on a Galaxy instance's database (ideally should be executed by a Galaxy admin). There are two methods in the script one each to generate two tabular files. The first file (`tool-popularity-20-04.tsv`) contains information about the usage of tools per month. The second file (`worflow-connection-20-04.tsv`) contains workflows present as the connections of tools. These tabular files are present under `/data` folder and can be used to run deep learning training by following steps.

2. Install the dependencies by executing the following lines if not done before:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_statistical_model`

3. Execute training script `train.sh`. Please check that the complete workflow file (`worflow-connection-20-04.tsv`) is being used in the training script.

The training script has following input parameters:

    `python <main python script> -wf <path to workflow file> -tu <path to tool usage file> -om <path to the final model file> -cd <cutoff date> -pl <maximum length of tool path>`
    
### Description of all parameters mentioned in the training script:

   - `<main python script>`: This script is the entry point of the entire analysis. It is present at `scripts/main.py`.
   - `<path to workflow file>`: It is a path to a tabular file containing Galaxy workflows. E.g. `data/worflow-connection-20-04.tsv`.
   - `<path to tool popularity file>`: It is a path to a tabular file containing usage frequencies of Galaxy tools. E.g. `data/tool-popularity-20-04.tsv`.
   - `<path to model file>`: It is a path of the final model (`h5` file). E.g. `data/tool_recommendation_model_statistical_model.hdf5`.
    
   - `<cutoff date>`: It is used to set the earliest date from which the usage frequencies of tools should be considered. The format of the date is YYYY-MM-DD. This date should be in the past. E.g. `2017-12-01`.
    
   - `<maximum length of tool path>`: This takes an integer and specifies the maximum size of a tool sequence extracted from any workflow. Any tool sequence of length larger than this number is not included in the dataset for training. E.g. `25`.

### An example command:
   
   `python scripts/main.py -wf data/worflow-connection-20-04.tsv -tu data/tool-popularity-20-04.tsv -om data/tool_recommendation_model_statistical_model.hdf5 -cd '2017-12-01' -pl 25`
