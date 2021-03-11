# This code writes down the solution for Task 1
import numpy as np
import scipy
from scipy.optimize import minimize
import string
import random
import matplotlib.pyplot as plt
from data_utils import generate_trajectory_param_pairs, Series_Dataset, obtain_tr_val_test_idx
from data_utils import get_dataloaders
from plot_utils import plot_trajectories, plot_losses
import pickle as pkl 
import os
import torch
import json
from models import RNN_model, train_rnn, evaluate_rnn
import argparse
from gs_utils import create_list_of_dicts

def create_and_save_dataset(N, num_trajs, num_realizations, filename, usenorm_flag=0):

    Z_pM = generate_trajectory_param_pairs(N, M=num_trajs, P=num_realizations, usenorm_flag=usenorm_flag)
    with open(filename, 'wb') as handle:
        pkl.dump(Z_pM, handle, protocol=pkl.HIGHEST_PROTOCOL)

def load_saved_dataset(filename):

    with open(filename, 'rb') as handle:
        Z_pM = pkl.load(handle)
    return Z_pM

class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python train.py --mode [train/test]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    #parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    #parser.add_argument("--model_file_saved", help="Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str)
    #parser.add_argument("--epoch_test", help="Particualr epoch to be tested on", type=int, default=None)
    parser.add_argument("--use_norm", help="Use_normalization", type=int, default=None)
    args = parser.parse_args() 
    #mode = args.mode
    model_type = args.model_type
    #epoch_test = args.epoch_test
    usenorm_flag = args.use_norm
    #model_file_saved = args.model_file_saved

    # Define the parameters of the model
    N = 1000

    # Plot the trajectory versus sample points
    num_trajs = 100
    num_realizations = 50

    if usenorm_flag == 1:
        datafile = "./data/trajectories_data_vars_normalized.pkl"
    else:
        datafile = "./data/trajectories_data_vars.pkl"

    if not os.path.isfile(datafile):
        print("Creating the data file: {}".format(datafile))
        create_and_save_dataset(N=N, num_trajs=num_trajs, num_realizations=num_realizations, 
                                filename=datafile, usenorm=usenorm_flag)

        Z_pM = load_saved_dataset(filename=datafile)
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:
        print("File already present")
        Z_pM = load_saved_dataset(filename=datafile)
    
    Z_pM_dataset = Series_Dataset(Z_pM_dict=Z_pM)

    tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_pM_dataset,
                                                                tr_to_test_split=0.9,
                                                                tr_to_val_split=0.833)
    print(len(tr_indices), len(val_indices), len(test_indices))

    batch_size = 128
    train_loader, val_loader, test_loader = get_dataloaders(Z_pM_dataset, batch_size, tr_indices, val_indices, test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    #for i_batch, sample_batched in enumerate(train_loader):
    #    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    #model_type = "gru"
    with open("configurations_var.json") as f:
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    # Json file to store grid search results
    jsonfile = './log/grid_search_results_{}_var.json'.format(model_type)
    
    # Parameters to be tuned
    gs_params = {
                "n_hidden":[20, 30, 40, 50, 60],
                "n_layers":[1, 2],
                "num_epochs":[2000]
                }
    
    # Creates the list of param combinations (options) based on the provided 'model_type'
    gs_list_of_options = create_list_of_dicts(options=options,
                                            model_type=model_type,
                                            param_dict=gs_params)
        
    print("Grid Search to be carried over following {} configs:\n".format(len(gs_list_of_options)))
    val_errors_list = []

    for i, gs_option in enumerate(gs_list_of_options):
        
        # Load the model with the corresponding options
        model = RNN_model(**gs_option)
    
        tr_verbose = True 
        save_chkpoints = False
        tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model = train_rnn(options=gs_option, 
                                                                                            nepochs=gs_option["num_epochs"],
                                                                                            train_loader=train_loader,
                                                                                            val_loader=val_loader,
                                                                                            device=device,
                                                                                            usenorm_flag=usenorm_flag,
                                                                                            tr_verbose=tr_verbose,
                                                                                            save_chkpoints=save_chkpoints)
        
        gs_option["Config_no"] = i+1
        gs_option["tr_loss_end"] = val_losses[-1]
        gs_option["val_loss_end"] = tr_losses[-1]
        gs_option["tr_loss_best"] = tr_loss_for_best_val_loss
        gs_option["val_loss_best"] = best_val_loss

        val_errors_list.append(gs_option)
        
    with open(jsonfile, 'w') as f:
        f.write(json.dumps(val_errors_list, indent=2))

    return None


if __name__ == "__main__":
    main()

    
