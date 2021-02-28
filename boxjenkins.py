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

def create_and_save_dataset(N, num_trajs, num_realizations, filename):

    Z_pM = generate_trajectory_param_pairs(N, M=num_trajs, P=num_realizations)
    with open(filename, 'wb') as handle:
        pkl.dump(Z_pM, handle, protocol=pkl.HIGHEST_PROTOCOL)

def load_saved_dataset(filename):

    with open(filename, 'rb') as handle:
        Z_pM = pkl.load(handle)
    return Z_pM

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python train.py --mode [train/test]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    parser.add_argument("--epoch_test", help="Particualr epoch to be tested on", type=int, default=None)
    args = parser.parse_args() 
    mode = args.mode
    model_type = args.model_type
    epoch_test = args.epoch_test

    # Define the parameters of the model
    N = 1000

    # Plot the trajectory versus sample points
    num_trajs = 100
    num_realizations = 10
    datafile = "./data/trajectories_data.pkl"

    if not os.path.isfile(datafile):
        print("Creating the data file")
        create_and_save_dataset(N=N, num_trajs=num_trajs, num_realizations=num_realizations, 
                                filename=datafile)

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
    with open("configurations.json") as f:
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    if mode.lower() == "train": 
        model_gru = RNN_model(**options[model_type])
        tr_verbose = True  
        tr_losses, val_losses, model = train_rnn(options=options[model_type], 
                                                nepochs=options[model_type]["num_epochs"],
                                                train_loader=train_loader,
                                                val_loader=val_loader,
                                                device=device,
                                                tr_verbose=tr_verbose)
        if tr_verbose == True:
            plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)

    elif mode.lower() == "test":

        model_file_saved = "./models/{}_ckpt_epoch_{}.pt".format(model_type, epoch_test)
        evaluate_rnn(options[model_type], test_loader, device, model_file=model_file_saved)

    return None


if __name__ == "__main__":
    main()

    