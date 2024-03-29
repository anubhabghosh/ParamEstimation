# This code writes down the solution for Task 1
# Creator: Anubhab Ghosh (anubhabg@kth.se)
# April 2022

import numpy as np
import scipy
from scipy.optimize import minimize
import string
import random
import matplotlib.pyplot as plt
from utils.data_utils import Series_Dataset, obtain_tr_val_test_idx, create_splits_file_name
from utils.data_utils import get_dataloaders, load_saved_dataset, NDArrayEncoder, load_splits_file
from utils.plot_utils import plot_trajectories, plot_losses
import pickle as pkl 
import os
import torch
import json
from src.rnn_models import RNN_model, train_rnn, evaluate_rnn
import argparse
from utils.gs_utils import create_list_of_dicts
from parse import parse

def check_if_dir_or_file_exists(file_path, file_name=None):
    flag_dir = os.path.exists(file_path)
    if not file_name is None:
        flag_file = os.path.isfile(os.path.join(file_path, file_name))
    else:
        flag_file = None
    return flag_dir, flag_file

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python train.py --mode [train/test]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")

    parser.add_argument("--data", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    parser.add_argument("--splits_file", help="Enter full path to splits file", type=str, default="splits_file.pkl")
    
    args = parser.parse_args() 
    datafile = args.data
    model_type = args.model_type
    splits_file = args.splits_file
    dataset_type, num_trajectories, num_realizations, N_seq = parse("ce_drive_trajectories_data_{}_M{:d}_P{:d}_N{:d}.pkl", datafile.split('/')[-1])
    datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))

    if splits_file is None or not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'create_function.py' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present")
        Z_pM = load_saved_dataset(filename=datafile)

    Z_pM_dataset = Series_Dataset(Z_pM_dict=Z_pM)

    #tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_pM_dataset,
    #                                                            tr_to_test_split=0.9,
    #                                                            tr_to_val_split=0.833)
    
    #print(len(tr_indices), len(val_indices), len(test_indices))

    if not os.path.isfile(splits_file):
        tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_pM_dataset,
                                                                    tr_to_test_split=0.9,
                                                                    tr_to_val_split=0.833)
        print(len(tr_indices), len(val_indices), len(test_indices))
        splits = {}
        splits["train"] = tr_indices
        splits["val"] = val_indices
        splits["test"] = test_indices
        splits_file_name = create_splits_file_name(dataset_filename=datafile,
                                                splits_filename=splits_file
                                                )
        with open(os.path.join(datafolder, splits_file_name), 'wb') as handle:
            pkl.dump(splits, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("Loading the splits file from {}".format(splits_file))
        splits = load_splits_file(splits_filename=splits_file)
        tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]


    batch_size = 128
    train_loader, val_loader, test_loader = get_dataloaders(Z_pM_dataset, batch_size, tr_indices, val_indices, test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    usenorm_flag = 0
    
    #with open("configurations.json") as f: # For estimating all theta parameters
    #    options = json.load(f)

    with open("./config/configurations_alltheta_ce_drive_prbs.json") as f: # Config file for estimating theta_vector when some parameters are fixed
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    # Json file to store grid search results
    # Define logfile paths and model file paths
    logfile_path = "./log/ce_drive/{}/{}_L{}_H{}_results/".format(dataset_type, 
                                                                    model_type, 
                                                                    options[model_type]["n_layers"], 
                                                                    options[model_type]["n_hidden"]
                                                                    )

    #modelfile_path = "./models/ce_drive/{}/{}_L{}_H{}_results/".format(dataset_type,
    #                                                                model_type, 
    #                                                                options[model_type]["n_layers"], 
    #                                                                options[model_type]["n_hidden"]
    #                                                                )

    json_file_name = 'grid_search_results_{}_M{}_P{}_N{}.json'.format(model_type,
                                                            num_trajectories,
                                                            num_realizations,
                                                            N_seq)

    log_file_name = "gs_training_{}_M{}_P{}_N{}.log".format(model_type,
                                                        num_trajectories,
                                                        num_realizations,
                                                        N_seq)

    #flag_models_dir, _ = check_if_dir_or_file_exists(modelfile_path,
    #                                                file_name=None)
    
    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(logfile_path,
                                                            file_name=log_file_name)
    
    flag_log_dir, flag_json_file = check_if_dir_or_file_exists(logfile_path,
                                                            file_name=json_file_name)
        
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    print("Is json-file present:? - {}".format(flag_log_file))
    
    #print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))
    
    if flag_log_dir == False:
        print("Creating {}".format(logfile_path))
        os.makedirs(logfile_path, exist_ok=True)
    
    #if flag_models_dir == False:
    #    print("Creating {}".format(modelfile_path))
    #    os.makedirs(modelfile_path, exist_ok=True)
    
    ####################################################################################
    # Parameters to be tuned
    gs_params = {
                "n_hidden":[30, 40, 50, 60], # "n_hidden":[20, 30, 40, 50, 60]
                "n_layers":[1, 2], # "n_layers":[1, 2],
                "num_epochs":[3000] # "n_epochs": [3000, 4000]
                }
    ####################################################################################

    # Creates the list of param combinations (options) based on the provided 'model_type'
    gs_list_of_options = create_list_of_dicts(options=options,
                                            model_type=model_type,
                                            param_dict=gs_params)
        
    print("Grid Search to be carried over following {} configs:\n".format(len(gs_list_of_options)))
    val_errors_list = []

    for i, gs_option in enumerate(gs_list_of_options):
        
        # Load the model with the corresponding options
        model = RNN_model(**gs_option)
    
        tr_verbose = True # Display training logs
        save_chkpoints = None # Don't save any checkpoints
        tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model = train_rnn(options=gs_option, 
                                                                                            nepochs=gs_option["num_epochs"],
                                                                                            train_loader=train_loader,
                                                                                            val_loader=val_loader,
                                                                                            device=device,
                                                                                            usenorm_flag=usenorm_flag,
                                                                                            tr_verbose=tr_verbose,
                                                                                            save_chkpoints=save_chkpoints,
                                                                                            logfile_path=os.path.join(logfile_path, log_file_name)
                                                                                            )
        
        gs_option["Config_no"] = i+1
        gs_option["tr_loss_end"] = tr_losses[-1]
        gs_option["val_loss_end"] = val_losses[-1]
        gs_option["tr_loss_best"] = tr_loss_for_best_val_loss
        gs_option["val_loss_best"] = best_val_loss

        val_errors_list.append(gs_option)
        
    with open(os.path.join(logfile_path, json_file_name), 'w') as f:
        f.write(json.dumps(val_errors_list, indent=2))

    return None


if __name__ == "__main__":
    main()

    
