# This file contains functions necessary for simulating the dataset parameters
import numpy as np
import string
import random
import torch
from torch.utils.data import Dataset, DataLoader

def generate_standard_U(N):

    u = np.random.uniform(0, 1, (N, 1))
    return u

def generate_standard_E(N):

    e = np.random.normal(0, 1, (N, 1))
    return e

def generate_param_realizations(d):

    theta = np.ones(shape=(d, 1))
    return theta

def create_input_matrix(u, d):

    U = np.zeros((u.shape[0], d))
    for i in range(d):
        
        v = np.roll(u, shift=i).flatten()
        for j in range(i):
            if not j is None:
                v[j] = 0
        U[:, i] = v

    return U

def generate_trajectory(N, theta_actual):

    # Creating the input matrix
    #u = generate_standard_U(N=N)
    #U = create_input_matrix(u, d)

    # Create the list of target parameter realizations
    d = 2

    # Create the measurement noise vector
    Y = np.empty((N, 1))
            
    # Creating the input matrix
    u = generate_standard_U(N=N)
    U = create_input_matrix(u, d)

    # For every trajectory this measurement noise is stored
    E_m = generate_standard_E(N)

    # Compute y_actual = U * theta + e
    Y = np.dot(U, theta_actual) + E_m

    return Y

def generate_trajectory_param_pairs(N=1000, M=50, P=5, usenorm_flag=0):

    # Define the parameters of the model
    #N = 1000

    # Plot the trajectory versus sample points
    #num_trajs = 5

    Z_pM = {}
    Z_pM["num_realizations"] = P
    Z_pM["num_trajectories"] = M
    Z_pM_data_lengths = []

    d = 2 # Only considering 2 dimensional data by default
    count = 0
    Z_pM_data = []

    for i in range(P):
        
        # Obtain a realization of theta
        theta_vector = generate_param_realizations(d=d)

        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            Y = generate_trajectory(N=N, theta_actual=theta_vector).reshape((-1, 1))
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

def main():

    num_trajs = 100
    num_realizations = 1
    usenorm_flag = 0
    N = 1000

    Z_pM = generate_trajectory_param_pairs(N, M=num_trajs, P=num_realizations, usenorm_flag=usenorm_flag)

if __name__ == "__main__":
    main()