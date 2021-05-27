import numpy as np
import torch
from data_utils import generate_trajectory_param_pairs, generate_trajectory_modified_param_pairs
import os
import pickle as pkl

def create_and_save_dataset(N, num_trajs, num_realizations, filename, usenorm_flag=0):

    #NOTE: Only variance estimation experiment
    # Currently this uses the 'modified' function
    # Otherwise use the 'usual' function 'generate_trajectory_variances_pairs'
    Z_pM = generate_trajectory_modified_param_pairs(N=N, 
                                                        M=num_trajs, 
                                                        P=num_realizations, 
                                                        usenorm_flag=usenorm_flag)

    # Saving the model
    with open(filename, 'wb') as handle:
        pkl.dump(Z_pM, handle, protocol=pkl.HIGHEST_PROTOCOL)

def main():

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
    N = 200
    # Number of trajectories (M)
    num_trajs = 500
    # Number of sample points (P)
    num_realizations = 50
    # Use normalization flag
    usenorm_flag = 0
    ######################################################

    if usenorm_flag == 1:
        #datafile = "./data/trajectories_data_vars_normalized.pkl"
        datafile = "./data/trajectories_data_normalized_modified_NS{}.pkl".format(
            int(num_trajs*num_realizations))
    else:
       # datafile = "./data/trajectories_data_vars.pkl"
       datafile = "./data/trajectories_data_modified_NS{}.pkl".format(
            int(num_trajs*num_realizations))

    if not os.path.isfile(datafile):

        print("Creating the data file: {}".format(datafile))
        create_and_save_dataset(N=N, num_trajs=num_trajs, num_realizations=num_realizations, 
                                filename=datafile, usenorm_flag=usenorm_flag)
    else:

        print("Dataset {} is already present!".format(datafile))
    
    print("Done...")
    return None

if __name__ == "__main__":
    main()
