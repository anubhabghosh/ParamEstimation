# Creator: Anubhab Ghosh (anubhabg@kth.se)
# April 2022

from utils.data_utils_simpler_model import generate_trajectory_param_pairs

import os
import pickle as pkl
import argparse
import numpy as np

def create_filename(P=50, M=500, N=200, use_norm=0, dataset_basepath="./data/", mode="all"):
    # Create the dataset based on the dataset parameters
    if use_norm == 1:
        #datafile = "trajectories_simpler_model_normalized_{}_M{}_P{}_N{}.pkl".format(mode, int(M), int(P), int(N))
        datafile = "trajectories_simpler_altmodel_normalized_{}_M{}_P{}_N{}.pkl".format(mode, int(M), int(P), int(N))
    else:
        #datafile = "trajectories_simpler_model_{}_M{}_P{}_N{}.pkl".format(mode, int(M), int(P), int(N))
        datafile = "trajectories_simpler_altmodel_{}_M{}_P{}_N{}.pkl".format(mode, int(M), int(P), int(N))

    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath

def save_dataset(Z_pM, filename):
    # Saving the dataset
    with open(filename, 'wb') as handle:
        pkl.dump(Z_pM, handle, protocol=pkl.HIGHEST_PROTOCOL)

def create_and_save_dataset(N, num_trajs, num_realizations, filename, usenorm_flag=0, mode="all"):
    
    np.random.seed(10) # This can be kept at a fixed step for being consistent
    if mode.lower() == "all":
        Z_pM = generate_trajectory_param_pairs(N=N, 
                                        M=num_trajs, 
                                        P=num_realizations, 
                                        usenorm_flag=usenorm_flag)
    
    save_dataset(Z_pM, filename=filename)

if __name__ == "__main__":
    
    usage = "Create datasets by simulating non-linear state space models \n"\
            "python create_dataset_simpler_model.py --num_realizations P --num_trajs M --sequence_length N --use_norm [0/1] --mode all --output_path [full path to output file]\n"\
            "Creates the dataset at the location output_path"\
        
    parser = argparse.ArgumentParser(description="Input arguments related to creating a synthetic dataset for a simpler model, for training RNN-based estimator")

    parser.add_argument("--num_realizations", help="denotes the number of realizations under consideration", type=int, default=50)
    parser.add_argument("--num_trajectories", help="denotes the number of trajectories to be simulated for each realization", type=int, default=500)
    parser.add_argument("--sequence_length", help="denotes the length of each trajectory", type=int, default=200)
    parser.add_argument("--use_norm", help="flag (0/1) to specify whether to use min-max normalization or not", type=int, default=0)
    #parser.add_argument("--mode", help="specify mode=pfixed (all theta, except theta_3, theta_4) / vars (variances) / all (full theta vector)", type=str, default=None)
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default=None)
    
    args = parser.parse_args() 

    N_seq = args.sequence_length
    num_trajectories = args.num_trajectories
    num_realizations = args.num_realizations
    usenorm_flag = args.use_norm
    #mode = args.mode
    output_path = args.output_path

    # Create the full path for the datafile
    datafile = create_filename(P=num_realizations, M=num_trajectories, N=N_seq, use_norm=usenorm_flag, 
                        dataset_basepath=output_path)

    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(datafile):
        print("Creating the data file: {}...".format(datafile))
        create_and_save_dataset(N=N_seq, num_trajs=num_trajectories, num_realizations=num_realizations,
                                filename=datafile, usenorm_flag=usenorm_flag)

    else:
        print("Dataset {} is already present!".format(datafile))
    
    print("Done!!")







