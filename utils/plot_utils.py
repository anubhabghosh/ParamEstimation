# This file contains functions for plotting the trajectories sampled
# Creator: Anubhab Ghosh (anubhabg@kth.se)
# April 2022

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(Z_pM, ncols=1, nrows=5):
    """ This function plots a few trajectories based on a particular sampled values of theta_i.
    For e.g. a single theta_i, gives a corresponding single realization of Y
    ---
    Args:
        - Z_pM ([dict]): A dictionary containing the details of the simulated trajectories - parameters
                        dataset.

        - ncols (int, optional): Number of columns in the subplot. Defaults to 1.
        - nrows (int, optional): Number of rows / trajectories to display in the subplot
    """

    p = Z_pM["num_realizations"]
    M = Z_pM["num_trajectories"] 
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle('Evolution of trajectories versus sample points')

    for i_row in range(nrows): 
    
        # Obtain a realization of theta
        i = np.random.randint(0, int(p*M), 1)

        theta_vector = Z_pM["data"][i][:, 0][-1] # Get the parameter vector
        Y = Z_pM["data"][i][:, 1][-1] # Get the trajectory
        N_y = len(np.ravel(Y)) # Get the length of the trajectory

        # Plot the generated trajectory
        axs[i_row].plot(np.arange(N_y), Y.reshape((-1,)))
        axs[i_row].set_title('Trajectory - {}'.format(i_row))
    
    plt.show()

# Plot the losses for training and validation set
def plot_losses(tr_losses, val_losses, logscale=True):
    """ This function plots the training and the validation 
    losses, with an option to plot the error in 'logscale' or
    'linear' scale
    """
    plt.figure(figsize=(10,5))
    if logscale == False:

        plt.plot(tr_losses, 'r+-')
        plt.plot(val_losses, 'b*-')
        plt.xlabel("No. of training iterations", fontsize=16)
        plt.ylabel("MSE Loss", fontsize=16)
        plt.legend(['Training Set', 'Validation Set'], fontsize=16)
        plt.title("MSE loss vs. no. of training iterations", fontsize=20)

    elif logscale == True:

        plt.plot(np.log10(tr_losses), 'r+-')
        plt.plot(np.log10(val_losses), 'b*-')
        plt.xlabel("No. of training iterations", fontsize=16)
        plt.ylabel("Log of MSE Loss", fontsize=16)
        plt.legend(['Training Set', 'Validation Set'], fontsize=16)
        plt.title("Log of MSE loss vs. no. of training iterations", fontsize=20)
