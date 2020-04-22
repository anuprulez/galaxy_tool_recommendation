# Tool recommender system in Galaxy by storing sequences of tools

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

1. Execute data extraction script `extract_data.sh` to extract two tabular files - `tool-popularity-19-09.tsv` and `workflow-connections-19-09.tsv`. This script should be executed on a Galaxy instance's database (ideally should be executed by a Galaxy admin). There are two methods in the script one each to generate two tabular files. The first file (`tool-popularity-19-09.tsv`) contains information about the usage of tools per month. The second file (`workflow-connections-19-09.tsv`) contains workflows present as the connections of tools. Save these tabular files. These tabular files are present under `/data` folder and can be used to run deep learning training by following steps.

2. Install the dependencies by executing the following lines if not done before:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_statistical_model`

3. Execute training script `train.sh`. Please check that the complete workflow file (`workflow-connections-19-09.tsv`) is being used in the training script.

The training script has following input parameters:

    `python <main python script> -wf <path to workflow file> -tu <path to tool usage file> -om <path to the final model file> -cd <cutoff date> -pl <maximum length of tool path>`
    
### Description of all parameters mentioned in the training script:

   - `<main python script>`: This script is the entry point of the entire analysis. It is present at `scripts/main.py`.
   - `<path to workflow file>`: It is a path to a tabular file containing Galaxy workflows. E.g. `data/workflow-connections-19-09.tsv`.
   - `<path to tool popularity file>`: It is a path to a tabular file containing usage frequencies of Galaxy tools. E.g. `data/tool-popularity-19-09.tsv`.
   - `<path to model file>`: It is a path of the final model (`h5` file). E.g. `data/tool_recommendation_model_statistical_model.hdf5`.
    
   - `<cutoff date>`: It is used to set the earliest date from which the usage frequencies of tools should be considered. The format of the date is YYYY-MM-DD. This date should be in the past. E.g. `2017-12-01`.
    
   - `<maximum length of tool path>`: This takes an integer and specifies the maximum size of a tool sequence extracted from any workflow. Any tool sequence of length larger than this number is not included in the dataset for training. E.g. `25`.

### An example command:
   
   `python scripts/main.py -wf data/workflow-connections-19-09.tsv -tu data/tool-popularity-19-09.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25`

4. The training of the neural network takes a long time (> 24 hours) for the complete data. Once the script finishes, `h5` model file is created at the given location (`path to model file`).

## The following steps are only necessary for deploying on Galaxy server.

5. Upload new model to: https://github.com/anuprulez/download_store/tree/tool_recommendation_model/tool_recommendation_model. 

6. In the `galaxy.yml.sample` config file, make the following changes:
    - Enable and then set the property `enable_tool_recommendations` to `true`.
    - Enable and then set the property `tool_recommendation_model_path` to `https://github.com/anuprulez/download_store/tree/tool_recommendation_model/tool_recommendation_model`.

7. Now go to the workflow editor and choose any tool from the toolbox. Then, you can see a `right-arrow` in top-right of the tool. Click on it to see the recommended tools to be used after the previously chosen tool.
