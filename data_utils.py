# This file contains functions necessary for simulating the dataset parameters
import numpy as np
import string
import random
import torch
import json
from torch.utils.data import Dataset, DataLoader
import pickle as pkl

def generate_uniform(N, a, b):
    
    # theta = U(a, b)
    theta = np.random.uniform(low=a, high=b, size=(N, 1))
    return theta

def generate_normal(N, mean, std):

    # n = N(mean, std**2)
    n = np.random.normal(loc=mean, scale=std, size=(N,1))
    return n

def generate_driving_noise(k, a=1.2):
    
    #u_k = np.cos(a*k) # Previous idea (considering start at k=0)
    u_k = np.cos(a*(k+1)) # Current modification (considering start at k=1)
    return u_k

def generate_trajectory(N, theta_vector):
    """ This function generates the trajectory for a given value of N and 
    the theta vector 

    Assuming the vector is:
        theta = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7]^T

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
    v_k_arr = generate_normal(N=N, mean=0, std=np.sqrt(theta_vector[5]))
    e_k_arr = generate_normal(N=N, mean=0, std=np.sqrt(theta_vector[6]))

    # Initiating the recursion
    for k in range(N):

        # Generate driving noise (which is time varying)
        # Driving noise should be carefully selected as per value of k (start from k=0 or =1)
        u_k = generate_driving_noise(k, a=1.2) 

        # For each instant k, sample v_k, e_k
        v_k = v_k_arr[k]
        e_k = e_k_arr[k]
        
        #v_k = generate_normal(N=1, mean=0, std=theta_vector[5])
        #e_k = generate_normal(N=1, mean=0, std=theta_vector[6])

        # Equation for updating the hidden state
        x[k+1] = x[k] * theta_vector[0] + (x[k] / (x[k]**2 + theta_vector[2])) * theta_vector[1] + \
                 u_k * theta_vector[3] + v_k
        
        # Equation for calculating the output state
        y[k] = theta_vector[4] * (x[k]**2) + e_k
    
    return y

def sample_parameter():
    """ Heuristically define the parameter sampling
    as per question provided

    Returns:
        theta: Parameter vector
    """
    theta_1 = generate_uniform(N=1, a=0, b=0.9) # Parameter a
    theta_2 = generate_uniform(N=1, a=20, b=30) # Parameter b
    theta_3 = generate_uniform(N=1, a=0.1, b=2) # Parameter c (often fixed as 1)
    theta_4 = generate_uniform(N=1, a=1, b=10) # Parameter d (often fixed as 8, as not globally identifiable)
    eps = np.finfo(float).eps # Get the machine epsilon 
    theta_5 = generate_uniform(N=1, a=0.01, b=1) # Parameter e
    theta_6 = generate_uniform(N=1, a=eps, b=10) # Variance Q
    theta_7 = generate_uniform(N=1, a=eps, b=5) # Variance R

    theta_vector = np.array([theta_1,
                    theta_2,
                    theta_3,
                    theta_4,
                    theta_5,
                    theta_6,
                    theta_7]).reshape((7, 1))

    #TODO: Need to fix this, inverse of mean is not a good idea, rather use inverse of variances (from Uniform distribution)
    weight_vector = 1.0 / (np.array((np.array([0.9/2, 50.0/2, 2.1/2, 11.0/2, 1.01/2, (10+eps)/2, (5+eps)/2]))) + eps)

    return theta_vector, weight_vector

def sample_parameter_modified():
    """ Heuristically define the parameter sampling
    as per question provided

    Returns:
        theta: Parameter vector
    """
    theta_1 = generate_uniform(N=1, a=0, b=0.9) # Parameter a
    theta_2 = generate_uniform(N=1, a=20, b=30) # Parameter b
    theta_3 = generate_uniform(N=1, a=0.1, b=1.75) # Parameter c (often fixed as 1)
    theta_4 = generate_uniform(N=1, a=4, b=12) # Parameter d (often fixed as 8, as not globally identifiable)
    eps = np.finfo(float).eps # Get the machine epsilon 
    theta_5 = generate_uniform(N=1, a=0.01, b=1) # Parameter e
    theta_6 = generate_uniform(N=1, a=0.1, b=1.5) # Variance Q
    theta_7 = generate_uniform(N=1, a=eps, b=1) # Variance R

    theta_vector = np.array([theta_1,
                    theta_2,
                    theta_3,
                    theta_4,
                    theta_5,
                    theta_6,
                    theta_7]).reshape((7, 1))

    # Hardcoding the weight vector values as ranges are known and pre-defined
    # Weight vector values are the inverse of the variances 
    # (inverse of uniformly distributed variances)
    weight_vector = np.array((14.81, 0.12, 4,408, 0.1875, 12.244, 6.122, 12.0))

    return theta_vector, weight_vector

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
        theta_vector, _ = sample_parameter()
        
        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            Y = generate_trajectory(N=N, theta_vector=theta_vector).reshape((-1, 1))
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

def generate_trajectory_variances_pairs(N=1000, M=50, P=5, usenorm_flag=0):

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
        variances_vector = sample_parameter()[0][-2:].reshape((2, 1))  # Choose only the last two parameters (as stochastically generated)
        fixed_theta_vector = np.array([0.5, 25, 1, 8, 0.05]).reshape((5,1)) # Fixed parameters
        # Combine them to form the theta vector required for generating trajectories
        theta_vector = np.concatenate((fixed_theta_vector, variances_vector), axis=0) 

        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            Y = generate_trajectory(N=N, theta_vector=theta_vector).reshape((-1, 1))
            # Normalize the data in range [0,1]
            if usenorm_flag == 1:
                Y = normalize(Y, feature_space=(0,1))
            elif usenorm_flag == 0:
                pass
            Z_pM_data.append([variances_vector, Y])
            Z_pM_data_lengths.append(N) 
        
    Z_pM["data"] = np.row_stack(Z_pM_data).astype(np.object)
    #Z_pM["data"] = Z_pM_data
    Z_pM["trajectory_lengths"] = np.vstack(Z_pM_data_lengths)

    return Z_pM

def generate_trajectory_modified_variances_pairs(N=1000, M=50, P=5, usenorm_flag=0):

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
        variances_vector = sample_parameter_modified()[0][-2:].reshape((2, 1))  # Choose only the last two parameters (as stochastically generated)
        fixed_theta_vector = np.array([0.5, 25, 1, 8, 0.05]).reshape((5,1)) # Fixed parameters
        # Combine them to form the theta vector required for generating trajectories
        theta_vector = np.concatenate((fixed_theta_vector, variances_vector), axis=0) 

        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            Y = generate_trajectory(N=N, theta_vector=theta_vector).reshape((-1, 1))
            # Normalize the data in range [0,1]
            if usenorm_flag == 1:
                Y = normalize(Y, feature_space=(0,1))
            elif usenorm_flag == 0:
                pass
            Z_pM_data.append([variances_vector, Y])
            Z_pM_data_lengths.append(N) 
        
    Z_pM["data"] = np.row_stack(Z_pM_data).astype(np.object)
    #Z_pM["data"] = Z_pM_data
    Z_pM["trajectory_lengths"] = np.vstack(Z_pM_data_lengths)

    return Z_pM

def generate_trajectory_fixed_variances_pairs(N=1000, M=50, P=5, usenorm_flag=0):

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
        fixed_variances_vector = np.array([1, 0.1]).reshape((2, 1))  # Choose only the last two parameters (as fixed)
        fixed_theta_vector = np.array([0.5, 25, 1, 8, 0.05]).reshape((5,1)) # Fixed parameters
        # Combine them to form the theta vector required for generating trajectories
        theta_vector = np.concatenate((fixed_theta_vector, fixed_variances_vector), axis=0) 

        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            Y = generate_trajectory(N=N, theta_vector=theta_vector).reshape((-1, 1))
            # Normalize the data in range [0,1]
            if usenorm_flag == 1:
                Y = normalize(Y, feature_space=(0,1))
            elif usenorm_flag == 0:
                pass
            Z_pM_data.append([variances_vector, Y])
            Z_pM_data_lengths.append(N) 
        
    Z_pM["data"] = np.row_stack(Z_pM_data).astype(np.object)
    #Z_pM["data"] = Z_pM_data
    Z_pM["trajectory_lengths"] = np.vstack(Z_pM_data_lengths)

    return Z_pM

def generate_trajectory_fixed_param_pairs(N=1000, M=50, P=5, usenorm_flag=0):

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
        
        # Use a fixed realization of theta
        theta_vector = np.array([0.5, 25, 1, 8, 0.05, 1, 0.1]).reshape((-1, 1))
        
        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            Y = generate_trajectory(N=N, theta_vector=theta_vector).reshape((-1, 1))
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
