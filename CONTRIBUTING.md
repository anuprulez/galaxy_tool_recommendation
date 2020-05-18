# Contributing to tool recommender system in Galaxy using deep learning (Gated recurrent units neural network)

Following steps can be followed to start contributing to this project:

1. Fork this repository (https://github.com/anuprulez/galaxy_tool_recommendation).
2. Create a new branch.
3. Install the dependencies by executing the following lines:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_gru_wc`
4. The scripts are located at `scripts/`.
5. Data files is located at `data/`.
6. Add new features/techniques.
7. Run the project using `sh train.sh`.
    - To run this project on complete set of workflow, large compute resource is needed (with at least 20-30 GB RAM) and running time is > 24 hours.
    - Details of the parameters in the training script are given in `README.md`.
8. Get a recommended model at `data/<<file name>>.hdf5`.
9. See recommended tools using `ipython_script/tool_recommendation_gru_wc.ipynb` or place the newly created recommendation model (from step 7) at `ipython_script/data/<<file name>>.hdf5`.
10. Open a pull request against the main repository (https://github.com/anuprulez/galaxy_tool_recommendation).

## Contributors
1. Anup Kumar (https://github.com/anuprulez) (Main contributor).
2. Helena Rasche (https://github.com/hexylena) (Contributed to the scripts for data collection from Galaxy EU server and to the Galaxy tool prediction API).
3. ...
