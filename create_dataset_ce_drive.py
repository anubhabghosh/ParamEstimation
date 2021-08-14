import argparse
from ast import parse
import pickle as pkl
import numpy as np
import os
from utils.ce_drive_data_utils import generate_trajectory_param_pairs_ce_drive
from utils.ce_drive_get_params import get_prbs_dataset, get_uniform_dataset

def create_filename(P=50, M=500, N=200, use_norm=0, dataset_basepath="./data/", dataset_type="prbs"):
    # Create the dataset based on the dataset parameters
    if use_norm == 1:
        datafile = "ce_drive_trajectories_data_normalized_{}_M{}_P{}_N{}.pkl".format(dataset_type, int(M), int(P), int(N))
    else:
        datafile = "ce_drive_trajectories_data_{}_M{}_P{}_N{}.pkl".format(dataset_type, int(M), int(P), int(N))

    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath

def save_dataset(Z_pM, filename):
    # Saving the dataset
    with open(filename, 'wb') as handle:
        pkl.dump(Z_pM, handle, protocol=pkl.HIGHEST_PROTOCOL)

def create_and_save_dataset(u, N, num_trajs, num_realizations, filename, Ts, 
    usenorm_flag=0, input_params_dict_file="./data/coupled_drive/prbs_dataset.json"):

    Z_pM = generate_trajectory_param_pairs_ce_drive(u=u,
                                                    N=N, 
                                                    M=num_trajs, 
                                                    P=num_realizations, 
                                                    usenorm_flag=usenorm_flag,
                                                    Ts=Ts,
                                                    input_params_dict=input_params_dict_file)
    
    save_dataset(Z_pM, filename=filename)
    
def main():

    usage = "Create datasets by simulating coupled-electric drive models \n"\
            "python create_dataset_ce_drive.py --num_realizations P --num_trajectories M --sequence_length N --use_norm [0/1]\n \
            --input_signal_type <Type of input signal <prbs/uniform>\
            --input_signal_path <Path to the input signal <prbs/uniform> \
            --output_path <Path to where you want to store the dataset>" \
            "Creates the dataset at the location output_path"\
        
    parser = argparse.ArgumentParser(description="Description of input arguments related to creating a dataset for training RNNs")

    parser.add_argument("--num_realizations", help="denotes the number of realizations under consideration", type=int, default=50)
    parser.add_argument("--num_trajectories", help="denotes the number of trajectories to be simulated for each realization", type=int, default=500)
    parser.add_argument("--sequence_length", help="denotes the length of each trajectory", type=int, default=200)
    parser.add_argument("--use_norm", help="flag (0/1) to specify whether to use min-max normalization or not", type=int, default=0)
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default=None)
    parser.add_argument("--input_signal_path", help="Enter full path to the data file", type=str, default=None)
    parser.add_argument("--input_signal_type", help="Enter type of the data file (PRBS/UNIFORM)", type=str, default=None)
    parser.add_argument("--input_params_dict_file", help="Enter the full path (incl. filename) to the .json file containing the parameters \
                        use to sample realizations from", type=str, default="./data/coupled_drive/prbs_dataset_opt.json")

    args = parser.parse_args() 

    N_seq = args.sequence_length
    num_trajectories = args.num_trajectories
    num_realizations = args.num_realizations
    usenorm_flag = args.use_norm
    output_path = args.output_path
    input_signal_path = args.input_signal_path
    input_signal_type = args.input_signal_type
    input_params_dict_file = args.input_params_dict_file

    # Get the input signal
    if input_signal_type.lower() == "prbs":
        z1, u1, z2, u2, z3, u3, _, Ts, _ = get_prbs_dataset(dataset_path=input_signal_path)
        u = u1 # Assign the input signal as the first signal under PRBS dataset

    elif input_signal_type.lower() == "uniform":
        z11, u11, z12, u12, _, Ts, _ = get_uniform_dataset(dataset_path=input_signal_path)
        u = u11 # Assign the input signal as the first signal under UNIFORM dataset

    # Create the full path for the datafile
    datafile = create_filename(P=num_realizations, M=num_trajectories, N=N_seq, use_norm=usenorm_flag, 
                        dataset_basepath=output_path, dataset_type=input_signal_type)

    #input_params_dict_file = os.path.join(output_path, "{}_dataset.json".format(input_signal_type))

    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(datafile):
        print("Creating the data file: {}".format(datafile))
        create_and_save_dataset(u=u, N=N_seq, num_trajs=num_trajectories, 
                                num_realizations=num_realizations, filename=datafile, Ts=Ts, 
                                usenorm_flag=usenorm_flag, input_params_dict_file=input_params_dict_file)

    else:
        print("Dataset {} is already present!".format(datafile))
    
    print("Done...")
    return None

if __name__ == "__main__":
    main()