# DeepBayes

An estimator for parameter estimation in stochastic nonlinear dynamical models by leveraging deep recurrent neural networks

## Dependencies

### PSEM, CME-MH, CPF-SAEM (MATLAB Code)

MATLAB 2020a with plug-ins such as: 

- System identification toolbox 
- Control systems toolbox

### DeepBayes Code (in Python)
```
python==3.6.8
pytorch==1.6.0
numpy==1.21.5
scipy==1.7.3
control==0.9.0
matplotlib==3.4.3
```
## Repository organization

- `/data/` : Folder that should contains the training data, viz. simulated trajectories for a chosen value of P, M (P - number of parameter realizations, M -  number of simulated trajectories per realization).

- `/src/`: Folder that contains essential scripts containing model descriptions such as 
  - `rnn_models_NLSS_modified.py` : RNN model defined for the NLSS models.
  - `rnn_models_ce_drive.py` :  RNN model defined for the real-world  [coupled electric drive model](https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives).

- `/utils/`: This folder contains help scripts necessary to train the RNNs, help in generate training datasets, etc.
- `/config/`: This folder contains configurations for training RNN models in `.json` format. 
- `/log/`:  This folder contains the logfiles related to the training of the RNN models.
- `/models/`: This folder contains the trained model checkpoints saved as `*.pt` files.
- `/MATLAB_Code/`: This folder contains relevant code to simulate the MATLAB models.

## Dataset creation

 - `create_dataset_ce_drive.py` : This script generates the training data for the coupled-electric drive model
 - `create_dataset_revised.py`: This script generates the training data for NLSS models - depending on whether we are estimating variance parameters or some subset of the total number of parameters 
 - `create_dataset_simpler_model.py`: The same kind of script as `create_dataset_revised.py` but for the simpler version of the NLSS model used in the experimental setup. 

## Running scripts

The scripts with the name `main...` are used for starting the model training. Depending on the type of model required to be executed, the specific `main`file needs to be called. Options need to be provided while executing the scripts as command line arguments. To see list of available options prior to execution, use the option `--help` as a command line argument. E.g. to find the list of available options for the running script `main.py`, you can use:

```
python3 main.py --help
```

- `main.py` : Running the program for NLSS models for estimation of a subset of the unknown parameters
- `main_with_grid_search.py`: Running the program to conduct grid search for the RNN architecture (GRU / LSTM) for estimation of a subset of the unknown parameters
- `main_for_ce_drive.py` : Running the program for the coupled electric drive model  
- `main_for_ce_drive_with_grid_search.py` :  Running the program to conduct grid search for the RNN architecture (GRU / LSTM) for the coupled electric drive model  
- `main_for_var_estimate.py` : Running the program for NLSS models for estimation of the variance parameters
- `main_for_var_estimate_grid_search.py` : Running the program to conduct grid search for the RNN architecture (GRU / LSTM) for estimation of the variance parameters

## Analysis of results

We can see the analysed results in the notebooks under `/analysis_ipynbs/`

More information to be added soon. 