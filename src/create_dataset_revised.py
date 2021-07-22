import numpy as np
import torch
from utils.data_utils import generate_trajectory_modified_param_pairs, \
                       generate_trajectory_modified_variances_pairs, \
                       generate_trajectory_partialfixed_param_pairs
import os
import pickle as pkl
import argparse

def create_filename(P=50, M=500, N=200, use_norm=0, dataset_basepath="./data/", mode="pfixed"):
    # Create the dataset based on the dataset parameters
    if use_norm == 1:
        datafile = "trajectories_data_normalized_{}_M{}_P{}_N{}.pkl".format(mode, int(M), int(P), int(N))
    else:
        datafile = "trajectories_data_{}_M{}_P{}_N{}.pkl".format(mode, int(M), int(P), int(N))

    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath

def save_dataset(Z_pM, filename):
    # Saving the dataset
    with open(filename, 'wb') as handle:
        pkl.dump(Z_pM, handle, protocol=pkl.HIGHEST_PROTOCOL)

def create_and_save_dataset(N, num_trajs, num_realizations, filename, usenorm_flag=0, mode="pfixed"):

    #NOTE: Generates for pfixed theta estimation experiment
    # Currently this uses the 'modified' function
    #Z_pM = generate_trajectory_modified_param_pairs(N=N, 
    #                                                M=num_trajs, 
    #                                                P=num_realizations, 
    #                                                usenorm_flag=usenorm_flag)
    if mode == "pfixed":
        Z_pM = generate_trajectory_partialfixed_param_pairs(N=N, 
                                                            M=num_trajs, 
                                                            P=num_realizations, 
                                                            usenorm_flag=usenorm_flag)
    elif mode == "vars":
        Z_pM = generate_trajectory_modified_variances_pairs(N=N, 
                                                        M=num_trajs, 
                                                        P=num_realizations, 
                                                        usenorm_flag=usenorm_flag)
    elif mode == "all":
        Z_pM = generate_trajectory_modified_param_pairs(N=N, 
                                                    M=num_trajs, 
                                                    P=num_realizations, 
                                                    usenorm_flag=usenorm_flag)
    
    save_dataset(Z_pM, filename=filename)

def main():    

    usage = "Create datasets by simulating non-linear state space models \n"\
            "python create_dataset_revised.py --num_realizations P --num_trajs M --sequence_length N --use_norm [0/1]\n"\
            "Creates the dataset at the location output_path"\
        
    parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for training RNNs")

    parser.add_argument("--num_realizations", help="denotes the number of realizations under consideration", type=int, default=50)
    parser.add_argument("--num_trajectories", help="denotes the number of trajectories to be simulated for each realization", type=int, default=500)
    parser.add_argument("--sequence_length", help="denotes the length of each trajectory", type=int, default=200)
    parser.add_argument("--use_norm", help="flag (0/1) to specify whether to use min-max normalization or not", type=int, default=0)
    parser.add_argument("--mode", help="specify mode=pfixed (all theta, except theta_3, theta_4) / vars (variances) / all (full theta vector)", type=str, default=None)
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default=None)
    
    args = parser.parse_args() 

    N_seq = args.sequence_length
    num_trajectories = args.num_trajectories
    num_realizations = args.num_realizations
    usenorm_flag = args.use_norm
    mode = args.mode
    output_path = args.output_path

    ######################################
    # Initial configuration
    ######################################
    # Length of each trajectory (N)
    #N = 1000
    # Number of trajectories (M)
    #num_trajs = 100
    # Number of sample points (P)
    #num_realizations = 100
    # Use normalization flag
    #usenorm_flag = 0
    ######################################

    #####################################################
    # Modified configuration (larger M, P and smaller N)
    #####################################################
    # Length of each trajectory (N)
    #N = 200
    # Number of trajectories (M)
    #num_trajectories = 500
    # Number of sample points (P)
    #num_realizations = 50
    # Use normalization flag
    #usenorm_flag = 0
    ######################################################

    # Create the full path for the datafile
    datafile = create_filename(P=num_realizations, M=num_trajectories, N=N_seq, use_norm=usenorm_flag, 
                        dataset_basepath=output_path, mode=mode)

    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(datafile):

        print("Creating the data file: {}".format(datafile))
        create_and_save_dataset(N=N_seq, num_trajs=num_trajectories, num_realizations=num_realizations,
                                filename=datafile, usenorm_flag=usenorm_flag, mode=mode)

    else:

        print("Dataset {} is already present!".format(datafile))
    
    print("Done...")
    return None

if __name__ == "__main__":
    main()







