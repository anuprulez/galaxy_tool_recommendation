# Description of output result files

Each of the 6 folders contain training and test results of 10 experiment runs of different neural network architectures to create tool recommendation model. 

## Neural network architectures/approaches used for comparison

  - CNN - Convolutional neural network with binary cross-entropy loss function.
  - CNN_WC - Convolutional neural network with weighted cross-entropy loss function.
  - DNN - Dense neural network with binary cross-entropy loss function.
  - DNN_WC - Dense neural network with weighted cross-entropy loss function.
  - GRU - Gated recurrent units neural network with binary cross-entropy loss function.
  - GRU_WC - Gated recurrent units neural network with weighted cross-entropy loss function.
  
Each neural network is executed for 10 times (10 experiment runs) with different hyperparameters (optimised by Bayesian optimisation) to compute average performance scores for precision, training loss, usage frequency (weights) and validation loss. All the result files are used to prepare two line plots as shown in paper.

