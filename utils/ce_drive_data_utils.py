# Creator: Anubhab Ghosh (anubhabg@kth.se)
# April 2022

# This file contains functions necessary for simulating the dataset parameters
import numpy as np
import string
import random
import torch
import json
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from .ce_drive_get_params import compute_actual_tf, simulate_model, isstable
from .data_utils import normalize
from control.matlab import c2d
import control as ct

def generate_uniform(N, a, b):
    """ This function defines the prior for sampling the parameters

    Args:
        N ([int]): The length of the vector 
        a ([float]): lower limit of the support
        b ([float]): upper limit of the support

    Returns:
        theta ([numpy.array]): The length of the parameter vector
    """
    # theta = U(a, b)
    theta = np.random.uniform(low=a, high=b, size=(N,1))
    return np.asscalar(theta)

def generate_normal(N, mean, std):
    """ This function defines the prior for sampling the parameters

    Args:
        N ([int]): The length of the vector 
        mean ([float]): mean of the normal distribution
        std ([float]): standard dev. of the normal distribution

    Returns:
        theta ([numpy.array]): The length of the parameter vector
    """
    # n = N(mean, std**2)
    n = np.random.normal(loc=mean, scale=std, size=(N,1))
    return n

def sample_parameter_modified(params_mean_std_dict):
    """ Heuristically define the parameter sampling
    as per question provided

    Returns:
        theta: Parameter vector
    """

    # Get the mean and standard deviation obtained from the parameters of the CT model:
    # k, \alpha, \omega0, \xi 
    k_mean = params_mean_std_dict["mean"]["k"]
    alpha_mean = params_mean_std_dict["mean"]["alpha"]
    omega0_mean = params_mean_std_dict["mean"]["omega0"]
    xi_mean = params_mean_std_dict["mean"]["xi"]

    percent_ = 0.2 # Percentage within mean (+ or - 20 %) parameter
    eps = np.finfo(float).eps # Get the machine epsilon 

    theta_1 = generate_uniform(N=1, a=k_mean * (1-percent_), b=k_mean * (1+percent_)) # Parameter k
    theta_2 = generate_uniform(N=1, a=alpha_mean * (1-percent_), b=alpha_mean  * (1+percent_)) # Parameter \alpha
    theta_3 = generate_uniform(N=1, a=omega0_mean * (1-percent_), b=omega0_mean  * (1+percent_)) # Parameter \omega0
    theta_4 = generate_uniform(N=1, a=xi_mean * (1-percent_), b=xi_mean  * (1+percent_)) # Parameter \xi
    theta_5 = generate_uniform(N=1, a=eps, b=0.01) # Parameter wn_variance (assuming noise is zero-mean)

    theta_vector = np.array([theta_1,
                    theta_2,
                    theta_3,
                    theta_4,
                    theta_5]).reshape((5, 1))

    return theta_vector

def generate_trajectory(N, u, Ts, theta_vector, add_noise_flag=False):
    """ This function generates the trajectory for a given value of N and 
    the theta vector 

    Assuming the vector is:
        theta = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7]^T

    Args:
        N ([int]): The length of the trajectory
        theta_vector ([type]): An estimate of the theta_vector (a single realization of Theta)
                            that remains the same during the recursion
    """
    # Since theta_5 is modeling variances in this code, 
    # for direct comparison with the MATLAB code, the std param input should be
    # a square root version
    e_k_arr = generate_normal(N=N, mean=0, std=np.sqrt(theta_vector[4]))

    Hs = compute_actual_tf(k=theta_vector[0], alpha=theta_vector[1], omega0=theta_vector[2],
                           xi=theta_vector[3])

    # Convert the transfer function of the continuous time model (CT) to discrete 
    # version through Euler discretization
    model_zoh_c2d= c2d(sysc=Hs, Ts=Ts, method='zoh') # Discretized version of CT model
    
    flag = isstable(np.array(ct.pole(model_zoh_c2d)), 
                sys_type='discrete')
    
    assert flag == True, "System is not stable!"

    y_model_zoh_c2d = simulate_model(Gs=model_zoh_c2d, u=u.reshape((-1,))).reshape((-1, 1)) # Output of Discretized model
    y = y_model_zoh_c2d + e_k_arr # Add the measurement noise

    return y

def generate_trajectory_param_pairs_ce_drive(u, N=1000, M=50, P=5, usenorm_flag=0, Ts=20e-3, 
                                            input_params_dict="./data/coupled_drive/prbs_dataset.json"):

    # Define the parameters of the model
    #N = 1000

    # Plot the trajectory versus sample points
    #num_trajs = 5
    with open(input_params_dict) as f:
        params_mean_std_dict = json.load(f)

    Z_pM = {}
    Z_pM["num_realizations"] = P
    Z_pM["num_trajectories"] = M
    Z_pM_data_lengths = []

    count = 0
    Z_pM_data = []

    for i in range(P):
        
        # Obtain a realization of theta
        flag = False # Set the stability-indicating flag to be False
        while flag == False:

            theta_vector = sample_parameter_modified(params_mean_std_dict)
            Hs = compute_actual_tf(k=theta_vector[0], alpha=theta_vector[1], omega0=theta_vector[2],
                                xi=theta_vector[3])

            # Convert the transfer function of the continuous time model (CT) to discrete 
            # version through Euler discretization
            model_zoh_c2d= c2d(sysc=Hs, Ts=Ts, method='zoh') # Discretized version of CT model
            flag = isstable(np.array(ct.pole(model_zoh_c2d)), 
                            sys_type='discrete')

        for m in range(M): 
            
            # Obtain the trajectory from the recursion
            Y = generate_trajectory(N=N,
                                    u=u, 
                                    Ts=Ts,
                                    theta_vector=theta_vector).reshape((-1, 1)
                                    )

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

