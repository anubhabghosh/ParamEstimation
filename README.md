# DeepBayes

An estimator for parameter estimation in stochastic nonlinear dynamical models by leveraging deep recurrent neural networks

## Dependencies

### PSEM, CME-MH, CPF-SAEM (MATLAB Code)

MATLAB 2020a with plug-ins such as: 

- [System identification toolbox](https://se.mathworks.com/products/sysid.html)
- [Control systems toolbox](https://se.mathworks.com/products/control.html)

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
- `/MATLAB_Code/`: This folder contains relevant code to simulate the MATLAB models. Relevant MATLAB code is found for:
  - `/cpf_saem/`: This folder contains code to execute the conditional CPF-SAEM method for estimating the two variance parameters, you should execute the file [`compute_ML_performance.m`](https://github.com/anubhabghosh/ParamEstimation/blob/master/MATLAB_Code/AlternativeML_based_methods/cpf_saem/compute_ML_performance.m).
  - `/SimplerNLSS_Model/`: This folder contains code to execute the PSEM method for the simpler NLSS model for estimating 4 unknown parameters, you should execute the file [`run_ML_based_fixed_theta_random_init.m`](https://github.com/anubhabghosh/ParamEstimation/blob/master/MATLAB_Code/AlternativeML_based_methods/NL_time_series_model/SimplerNLSSmodel/run_ML_based_fixed_theta_random_init.m)
  - `/Linear_Toy_model/` : This folder contains code for executing the linear estimator based on the approach of DeepBayes for the linear Toy model, you should run [`run_multiple_simulations_fixed_theta_w_bias.m`](https://github.com/anubhabghosh/ParamEstimation/blob/master/MATLAB_Code/Linear_Toy_Model/Linear_Model/run_multiple_simulations_fixed_theta_w_bias.m) to execute the method for a multiple values of P, M in order to replicate the table of varying MSE versus P, M.
- `/analysis_ipynbs/`: This folder contains IPython notebooks used for analysing the results.

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

## How to run the code for DeepBayes ...
1. Generate the appropriate dataset by calling one of the dataset creation files. List of available options that can be accessed are found by using
`python3 create_dataset_[revised/ce_drive/simpler_model].py --help`
    E.g. `create_dataset_revised.py` can be used to create the dataset for estimating variance as:
    ```
    python3 create_dataset_revised.py --num_realizations P --num_trajs M --sequence_length N --mode vars --use_norm [0/1 (Usually kept 0)] --output_path [output path to the .pkl file containing the dataset] 
    ```
 2. The configuration to run the DeepBayes method is available for edit (if necessary) in `/config/`. 
 3. Run the training algorithm by calling the appropriate main function, e.g. for `simpler_model` (estimating all parameters of the simpler model shown in the paper) use `main.py`, for variance estimation use `main_for_var_estimate.py` and for coupled drive use `main_for_ce_drive.py`. Before running the file check the values of the variables `logfile_path`, `modelfile_path`, `main_exp_name` and set your own path if required (for now these values are hardcoded). E.g. to run `main.py` for DeepBayes GRU for estimating all 4 parameters of the simpler NLSS model, one may need to use: 
    ```
    python3 main.py --mode train --model_type gru --dataset_mode pfixed --config [path to config file] --datafile [path to the dataset (.pkl file)] --use_norm 0 --splits [path to the splits file (splitting data into train, val, ...), for the first time this will be created]
    ```
 5. The log files will be stored under appropriate folders under `/log/` and model checkpoints under `/models/`. 
 6. Check one of the analysis IPython notebooks where you can simulate or load an evaluation dataset, load the saved model and perform inference. In some cases, it may ne required to run the MATLAB code for the comparing methods (like PSEM (NLSS Simpler Model), CPF_SAEM and CME-MH (Metropolis hastings) and then put the location to the results in the notebook.
 
