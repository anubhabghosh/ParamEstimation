###################################################################################
# This file contains functions necessary for simulating the dataset parameters
# Model definition:
# x_{k+1} &= \theta_{1} \frac{x_k}{x_{k}^{2} + 1} + u_{k} + \theta_{2} v_{k} \\
# y_{k} &= \theta_{3} x_{k}^{2} + \theta_{4} e_{k} \\
# where u_{k} = cos(1.2k), v_{k}, e_{k} \sim \mathcal{N}(0, 1), k=1, 2, ..., N. 
# Assuming v_{k} and e_{k} are independent of each other.
###################################################################################
import numpy as np
import string
import random
import torch
import json
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import os

#####################################################################################
# This function generates realizations from uniformly distributed prior distribuions
#####################################################################################
def generate_uniform(N, a, b):
    
    # theta = U[a, b]
    theta = np.random.uniform(low=a, high=b, size=(N, 1))
    return np.asscalar(theta)

#####################################################################################
# This function generates realizations from normally distributed distribuions for noise
#####################################################################################
def generate_normal(N, mean, std):

    # n = N(mean, std**2)
    n = np.random.normal(loc=mean, scale=std, size=(N,1))
    return n

#####################################################################################
# This function generates the driving signal for the state space equations
#####################################################################################
def generate_driving_signal(k, a=1.2, add_noise=False):
    
    if add_noise == False:
        u_k = np.cos(a*(k+1)) # Current modification (V.IMP.) (considering start at k=1)
    elif add_noise == True:
        u_k = np.cos(a*(k+1) + np.random.normal(loc=0, scale=np.pi, size=(1,1))) # Adding noise to the sample
    return u_k

################################################################################
# This function generates \theta vectors using specified values of \theta_{1},
# \theta_{2}, \theta_{3}, \theta_{4}
################################################################################
def sample_parameter():
    """ Heuristically define the parameter sampling
    as per question provided

    Returns:
        theta: Parameter vector
    """
    theta_1 = generate_uniform(N=1, a=0, b=1.0) # Parameter \theta_{1}
    theta_2 = generate_uniform(N=1, a=0.1, b=2) # Parameter \theta_{2}
    eps = np.finfo(float).eps # Get the machine epsilon 
    theta_3 = generate_uniform(N=1, a=eps, b=3) # Variance of v_{k}, i.e. \theta_{3}
    theta_4 = generate_uniform(N=1, a=eps, b=2) # Variance of e_{k}, i.e. \theta_{4}

    theta_vector = np.array([theta_1,
                    theta_2,
                    theta_3,
                    theta_4]).reshape((4, 1))

    return theta_vector

################################################################################
# This function generates \theta vectors using specified values of \theta_{1},
# \theta_{2}, \theta_{3}, \theta_{4} for the alternate model

# x_{k+1} = \theta_{1} * (x_{k} / ((0.2 * (x_{k}**2)) + 1)) + u_{k} + v_{k}
# y_{k} = \theta_{2} * x_{k} ** 2 + e_{k}
################################################################################
def sample_parameter_alternate_model():
    """ Heuristically define the parameter sampling
    as per the chosen priors for the alternative, simpler model provided. 
    We assume uniform priors for all the parameters present.

    Returns:
        theta: Parameter vector
    """
    theta_1 = generate_uniform(N=1, a=0, b=1.0) # Parameter \theta_{1}
    theta_2 = generate_uniform(N=1, a=0.1, b=2) # Parameter \theta_{2}
    eps = np.finfo(float).eps # Get the machine epsilon 
    theta_3 = generate_uniform(N=1, a=eps, b=1) # Variance of v_{k}, i.e. \theta_{3} 
    theta_4 = generate_uniform(N=1, a=eps, b=1) # Variance of e_{k}, i.e. \theta_{4}

    theta_vector = np.array([theta_1,
                    theta_2,
                    theta_3,
                    theta_4]).reshape((4, 1))

    return theta_vector

#########################################################################
# Define a function for generating trajectories for the simpler model
#########################################################################
def generate_trajectory(N, theta_vector, add_noise_flag=False):
    """ This function generates the trajectory for a given value of N and 
    the theta vector 

    Assuming the vector is:
        theta = [theta_1, theta_2, theta_3, theta_4]^T

    Args:
        N ([int]): The length of the trajectory
        theta_vector ([type]): An estimate of the theta_vector (a single realization of Theta)
                            that remains the same during the recursion
    """
    
    # Starting the recursion
    x = np.zeros((N+1,)) # Also includes initial zero value
    y = np.zeros((N,))

    #NOTE: Since theta_5 and theta_6 are modeling variances in this code, 
    # for direct comparison with the MATLAB code, the std param input should be
    # a square root version
    v_k_arr = generate_normal(N=N, mean=0, std=np.sqrt(theta_vector[2]))
    e_k_arr = generate_normal(N=N, mean=0, std=np.sqrt(theta_vector[3]))

    # Initiating the recursion
    for k in range(N):

        # Generate driving noise (which is time varying)
        # Driving noise should be carefully selected as per value of k (start from k=0 or =1)
        u_k = generate_driving_signal(k, a=1.2, add_noise=add_noise_flag) 

        # For each instant k, sample v_k, e_k
        v_k = v_k_arr[k]
        e_k = e_k_arr[k]
        
        # Equation for updating the hidden state
        x[k+1] = theta_vector[0] * (x[k] / (x[k]**2 + 1.0))  + u_k + v_k
        
        # Equation for calculating the output state
        y[k] = theta_vector[1] * (x[k]**2) + e_k
    
    return y

###################################################################################
# Define a function for generating trajectories for the alternative simpler model
###################################################################################
def generate_trajectory_alternative_model(N, theta_vector, add_noise_flag=False):
    """ This function generates the trajectory for a given value of N and 
    the theta vector for the alternative model

    Assuming the vector is:
        theta = [theta_1, theta_2, theta_3, theta_4]^T

    Args:
        N ([int]): The length of the trajectory
        theta_vector ([type]): An estimate of the theta_vector (a single realization of Theta)
                            that remains the same during the recursion
    """
    
    # Starting the recursion
    x = np.zeros((N+1,)) # Also includes initial zero value
    y = np.zeros((N,))

    #NOTE: Since theta_5 and theta_6 are modeling variances in this code, 
    # for direct comparison with the MATLAB code, the std param input should be
    # a square root version
    v_k_arr = generate_normal(N=N, mean=0, std=np.sqrt(theta_vector[2]))
    e_k_arr = generate_normal(N=N, mean=0, std=np.sqrt(theta_vector[3]))

    # Initiating the recursion
    for k in range(N):

        # Generate driving noise (which is time varying)
        # Driving noise should be carefully selected as per value of k (start from k=0 or =1)
        u_k = generate_driving_signal(k, a=1.2, add_noise=add_noise_flag)

        # For each instant k, sample v_k, e_k
        v_k = v_k_arr[k]
        e_k = e_k_arr[k]
        
        # Equation for updating the hidden state
        x[k+1] = theta_vector[0] * (x[k] / ((0.2*x[k])**2 + 1.0))  + u_k + v_k
        
        # Equation for calculating the output state
        y[k] = theta_vector[1] * (x[k]**2) + e_k
    
    return y

######################################################################################
# Normalize the generated trajectories through min-max scaling
######################################################################################
def normalize(X, feature_space=(0, 1)):
    """ Normalizing the features in the feature_space (lower_lim, upper_lim)

    Args:
        X ([numpy.ndarray]): Unnormalized data consisting of signal points
        feature_space (tuple, optional): [lower and upper limits]. Defaults to (0, 1).

    Returns:
        X_norm [numpy.ndarray]: Normalized feature values
    """
    X_normalized = (X - X.min())/(X.max() - X.min()) * (feature_space[1] - feature_space[0]) + \
        feature_space[0]
    return X_normalized

#################################################################################
# This function generates training data Z_{p,M} using specified values of p, M
# for the full set of \theta_{i}. theta_vectors are sampled from the priors defined 
# earlier 
#################################################################################
def generate_trajectory_param_pairs(N=1000, M=50, P=5, usenorm_flag=0):

    # Define the parameters of the model
    #N = 1000

    # Plot the trajectory versus sample points
    #num_trajs = 5

    Z_pM = {}
    Z_pM["num_realizations"] = P
    Z_pM["num_trajectories"] = M
    Z_pM_data_lengths = []

    count = 0
    Z_pM_data = []

    for i in range(P):
        
        # Obtain a realization of theta
        #theta_vector = sample_parameter()  # For existing, simpler model
        theta_vector = sample_parameter_alternate_model() # For alternative, simpler model
        
        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            #Y = generate_trajectory(N=N, theta_vector=theta_vector).reshape((-1, 1)) # for the existing, simpler model
            Y = generate_trajectory_alternative_model(N=N, theta_vector=theta_vector).reshape((-1, 1)) # for the alternative, simpler model
            # Normalize the data in range [0,1]
            if usenorm_flag == 1:
                Y = normalize(Y, feature_space=(0,1))
            elif usenorm_flag == 0:
                pass
            Z_pM_data.append([theta_vector, Y])
            Z_pM_data_lengths.append(N) 
        
    Z_pM["data"] = np.row_stack(Z_pM_data).astype(np.object)
    #Z_pM["data"] = Z_pM_data
    Z_pM["trajectory_lengths"] = np.vstack(Z_pM_data_lengths)

    return Z_pM

##########################################################################
# Utils for generating and loading datasets
##########################################################################
def load_splits_file(splits_filename):

    with open(splits_filename, 'rb') as handle:
        splits = pkl.load(handle)
    return splits

def load_saved_dataset(filename):

    with open(filename, 'rb') as handle:
        Z_pM = pkl.load(handle)
    return Z_pM

class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Series_Dataset(Dataset):

    def __init__(self, Z_pM_dict):

        self.data_dict = Z_pM_dict
        self.num_realizations = Z_pM_dict["num_realizations"]
        self.num_trajectories = Z_pM_dict["num_trajectories"]
        self.trajectory_lengths = Z_pM_dict["trajectory_lengths"]

    def __len__(self):

        return len(self.data_dict["data"])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.data_dict["data"][idx][1]

        sample = {"inputs": np.expand_dims(inputs, axis=0), 
                  "targets": self.data_dict["data"][idx][0]
                  }

        return sample

def obtain_tr_val_test_idx(dataset, tr_to_test_split=0.9, tr_to_val_split=0.83):

    num_training_plus_test_samples = len(dataset)
    print("Total number of samples: {}".format(num_training_plus_test_samples))
    print("Training + val to test split: {}".format(tr_to_test_split))
    print("Training to val split: {}".format(tr_to_val_split))

    num_train_plus_val_samples = int(tr_to_test_split * num_training_plus_test_samples)
    num_test_samples = num_training_plus_test_samples - num_train_plus_val_samples
    num_train_samples = int(tr_to_val_split * num_train_plus_val_samples)
    num_val_samples = num_train_plus_val_samples - num_train_samples
    
    indices = torch.randperm(num_training_plus_test_samples).tolist()
    tr_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:num_train_samples+num_val_samples]
    test_indices = indices[num_train_samples+num_val_samples:]

    return tr_indices, val_indices, test_indices

def my_collate_fn(batch):
    inputs = [item["inputs"] for item in batch]
    targets = [item["targets"] for item in batch]
    targets = torch.FloatTensor(targets)
    inputs = torch.from_numpy(np.row_stack(inputs))
    return (inputs, targets)

def get_dataloaders(dataset, batch_size, tr_indices, val_indices, test_indices=None):

    train_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=torch.utils.data.SubsetRandomSampler(tr_indices),
                            num_workers=0,
                            collate_fn=my_collate_fn)

    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
                            num_workers=0,
                            collate_fn=my_collate_fn)
    
    test_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=torch.utils.data.SubsetRandomSampler(test_indices),
                            num_workers=0,
                            collate_fn=my_collate_fn)

    return train_loader, val_loader, test_loader

def create_splits_file_name(dataset_filename, splits_filename):
    
    idx_dset_info = dataset_filename.rfind("M")
    idx_splitfilename = splits_filename.rfind(".pkl")
    splits_filename_modified = splits_filename[:idx_splitfilename] + "_" + dataset_filename[idx_dset_info:] 
    return splits_filename_modified

def create_file_paths(params_combination_list, filepath, main_exp_name):
    
    list_of_logfile_paths = []
    # Creating the logfiles
    for params in params_combination_list:

        exp_folder_name = "trajectories_M{}_P{}_N{}/".format(params["num_trajectories"],
                                                            params["num_realizations"],
                                                            params["N_seq"])

        #print(os.path.join(log_filepath, main_exp_name, exp_folder_name))
        full_path_exp_folder = os.path.join(filepath, main_exp_name, exp_folder_name)
        list_of_logfile_paths.append(full_path_exp_folder)
        os.makedirs(full_path_exp_folder, exist_ok=True)

    return list_of_logfile_paths

def get_list_of_config_files(model_type, options, dataset_mode='pfixed', params_combination_list=None, main_exp_name=None):
    
    #logfile_path = "./log/estimate_theta_{}/".format(dataset_mode)
    #modelfile_path = "./models/"
    if main_exp_name is None:
        main_exp_name = "{}_L{}_H{}_multiple".format(model_type, 
                                                     options[model_type]["n_layers"], 
                                                     options[model_type]["n_hidden"])
    else:
        pass

    base_config_dirname = os.path.dirname("./config/configurations_alltheta_pfixed.json")
    
    list_of_config_folder_paths = create_file_paths(params_combination_list=params_combination_list,
                                            filepath=base_config_dirname,
                                            main_exp_name=main_exp_name)

    #list_of_gs_jsonfile_paths = create_file_paths(params_combination_list=params_combination_list,
    #                                        filepath=modelfile_path,
    #                                        main_exp_name=main_exp_name)

    list_of_config_files = []
    
    for i, config_folder_path in enumerate(list_of_config_folder_paths):
        
        config_filename = "configurations_alltheta_pfixed_gru_M{}_P{}_N{}.json".format(
            params_combination_list[i]["num_trajectories"], params_combination_list[i]["num_realizations"], 
            params_combination_list[i]["N_seq"])
        os.makedirs(config_folder_path, exist_ok=True)
        config_file_name_full = os.path.join(config_folder_path, config_filename)
        list_of_config_files.append(config_file_name_full)
    
    # Print out the model files
    #print("Config files to be created at:")
    #print(list_of_config_files)
    
    return list_of_config_files

def check_if_dir_or_file_exists(file_path, file_name=None):
    flag_dir = os.path.exists(file_path)
    if not file_name is None:
        flag_file = os.path.isfile(os.path.join(file_path, file_name))
    else:
        flag_file = None
    return flag_dir, flag_file