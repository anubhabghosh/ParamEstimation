# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
import scipy
from scipy.io import loadmat, savemat
import control as ct
import os
from scipy.optimize import curve_fit
from control.matlab import tf, ss, lsim, bode, c2d
from .data_utils import NDArrayEncoder  # While running main program, uncomment this line
#from data_utils import NDArrayEncoder # While running only this file, comment the above import of same name, and uncomment this one
import json
from scipy import signal as sig

def generate_uniform(N, a, b):
    
    # theta = U(a, b)
    theta = np.random.uniform(low=a, high=b, size=(N,1))
    return theta.item()

def get_prbs_dataset(dataset_path=None):

    if dataset_path is None:
        dataset_path = './data/coupled_drive/DATAPRBS.MAT'
    else:
        pass

    data_prbs_signal = loadmat(dataset_path)
    # Extract input and output signals from the PRBS dataset
    z1, z2, z3 = data_prbs_signal['z1'], data_prbs_signal['z2'], data_prbs_signal['z3']
    u1, u2, u3 = data_prbs_signal['u1'], data_prbs_signal['u2'], data_prbs_signal['u3']

    N = len(z1)
    Ts = 20e-3

    # Model parameters a, b, c, d estimated for prbs input signal
    model_params_rsys_prbs = [{"a":5163, "b":-19.93, "c":-509.8, "d":-2835},
                            {"a":4015, "b":-17.96, "c":-463.7, "d":-2094},
                            {"a":5017, "b":-19.61, "c":-504.2, "d":-2747},
                            {"a":5174, "b":-20.52, "c":-512.5, "d":-2931},
                            {"a":25991, "b":-87.30, "c":-1703, "d":-16230}]

    return z1, u1, z2, u2, z3, u3, N, Ts, model_params_rsys_prbs

def get_uniform_dataset(dataset_path):

    if dataset_path is None:
        dataset_path = './data/coupled_drive/DATAUNIF.MAT'
    else:
        pass

    data_uniform_signal = loadmat(dataset_path)
    
    # Extract input and output signals from the uniform dataset
    z11, z12 = data_uniform_signal['z11'], data_uniform_signal['z12']
    u11, u12 = data_uniform_signal['u11'], data_uniform_signal['u12']

    N = len(z11)
    Ts = 20e-3

    # Model parameters a, b, c, d estimated for uniform input signal
    model_params_rsys_uniform = [{"a":-858.8, "b":-21.86, "c":-255.0, "d":-1079},
                                {"a":1531, "b":-25.08, "c":-406.2, "d":-1826},
                                {"a":2132, "b":-24.83, "c":-548.0, "d":-2609},
                                {"a":2567, "b":-23.93, "c":-627.4, "d":-3293},
                                {"a":2922, "b":-27.09, "c":-668.9, "d":-3834},
                                {"a":4186, "b":-41.64, "c":-870.8, "d":-5684},
                                {"a":10370, "b":-124.9, "c":-2694, "d":-23220}]

    return z11, u11, z12, u12, N, Ts, model_params_rsys_uniform

def plot_input_dataset(Ts, N, input_signal, output_signal):
    """ This function plots the input signal and the corresponding 
    output signal. The time scale (x-axis) is plotted using the parameters 
    Ts (sampling time) and N (number of pts.).
    ----
    Args:
        Ts ([float]): Sampling time (in s)
        N ([int]): Number of points
        input_signal ([numpy.array]): The input signal (driving signal such as 
                                        PRBS signal / uniform signal)
        output_signal ([numpy.array]): The output signal (measured response of the system)
    Returns:
        None
    """
    # Plot one of the datasets
    t = np.linspace(1, N, N)
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 22})
    plt.subplot(211)
    plt.plot(t*Ts, output_signal, label='Output signal')
    plt.grid()
    plt.legend()
    plt.subplot(212)
    plt.plot(t*Ts, input_signal, label='Input signal')
    plt.grid()
    plt.legend()
    plt.show()

def compute_model_tf(a, b, c, d):
    """ This function computes the transfer function Gs
    of the model
    ----
    Args:
        a ([float]): \theta_{0001} (numerator parameter)
        b ([float]): \theta_{0010} (coefficient of s**2 in denominator)
        c ([float]): \theta_{0100} (coefficient of s**1 in denominator)
        d ([float]): \theta_{1000} (coefficient of s**0 in denominator)

    Returns:
        Gs: transfer function ('s') of the model
    """
    s = tf('s')
    Gs = a/(s**3 -b*s**2 - c*s -d)
    #print(Gs)
    return Gs

def get_poles(b, c, d):
    """ This function gets the roots and real-root of a third order polynomial.
    Assumption: Polynomial is third-order 

    Args:
        b ([float]): coefficient of x**2
        c ([float]): coefficient of x**1
        d ([float]): coefficient of x**0

    Returns:
        alpha ([float]): Real root 
        poles ([numpy.array]): Roots of the third order polynomial, for real coefficients,
        this will give us three coefficients
    """
    poles = np.roots(np.array([1, -b, -c, -d]))
    alpha = (np.real(poles[np.isreal(poles)])).item()
    return alpha, poles

def compute_actual_parameters(a, b, c, d, alpha):
    """ Compute the actual parameters of 
    the H(s) transfer function (Eqn. (1)) of the paper
    ----
    Args:
        a ([float]): \theta_{0001} (numerator parameter)
        b ([float]): \theta_{0010} (coefficient of s**2 in denominator)
        c ([float]): \theta_{0100} (coefficient of s**1 in denominator)
        d ([float]): \theta_{1000} (coefficient of s**0 in denominator)
        alpha ([float]): negative of the \alpha parameter in H(s)

    Returns:
        k, alpha, omega0, xi
    """
    # Getting the parameters
    k = -a/d
    alpha = -alpha
    omega0 = np.sqrt(-d/alpha)
    xi = (-b - alpha)/(2*omega0)
    return k, alpha, omega0, xi

def compute_actual_tf(k, alpha, omega0, xi):
    """
    Computes the transfer function H(s) using the relevant parameters characterized in 
    Eqn. (1) of the coupled electric drives paper
    """
    s = tf('s')
    Hs = (k*alpha*omega0**2)/((s+alpha)*(s**2 + 2*xi*omega0*s + omega0**2))
    #print("Real part of the system poles:{}".format(np.real(ct.pole(Hs))))
    #print("System zeros:{}".format(ct.zero(Hs)))
    #print(Hs)
    flag = isstable(np.array(ct.pole(Hs)), sys_type='continuous')
    return Hs

def isstable(list_of_poles, sys_type='continuous'):
    """ This function checks whether a given model transfer function is
    stable or not. The type of check depends on whether the model is continuous
    and discrete.

    Args:
        list_of_poles ([numpy.array]): The poles of the transfer function
        sys_type (str, optional): [The type of the transfer function]. Defaults to 'continuous'.

    Returns:
        flag (bool): A flag to indicate stability of the transfer function. True = stable, False = unstable
    """
    if sys_type == "continuous":
        list_of_poles = np.real(list_of_poles)
        #print("Poles of the discrete system:{}".format(list_of_poles))
        if (list_of_poles < 0).all() == True:
            #print("Given {} system is stable!!".format(sys_type))
            flag = True
        else:
            #print("Given {} system is unstable!".format(sys_type))
            flag = False   

    elif sys_type == "discrete":
        #print("Poles of the discrete system:{}".format(np.abs(list_of_poles)))
        if (np.abs(list_of_poles) < 1).all() == True:
            #print("Given {} system is stable!!".format(sys_type))
            flag = True
        else:
            #print("Given {} system is unstable!".format(sys_type))
            flag = False   

    return flag

def simulate_bode(G):
    """ This function simulates the bode plot for a given transfer function
    ----
    Args:
        G ([s-function]): Transfer function
    """
    # Simulate the model Gs's bode plot
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 12})
    bode(G)
    plt.show()
    
def simulate_model(Gs, u, t_arr=None):
    """ This function simulates the model represented by Gs, 
    using the input signal u and time signal (t_arr), which 
    defines the number of time steps for which we should simulate the model

    Args:
        Gs ([s-function]): transfer function 
        u ([numpy.array]): input signal
        t_arr ([numpy.array], optional): time signal (t_arr)

    Returns:
        y_model ([numpy.array]): The output signal (which is the absolute value of the signal)
    """
    # Simulate output of a linear system
    if t_arr is None:
        y_model = lsim(sys=Gs, U=u)[0]
    else:
        y_model, T, xout = lsim(sys=ss(Gs), U=u, T=t_arr)
    y_model = np.abs(y_model)
    return y_model

def euler_discretize_model(a, b, c, d, Ts):
    """ This function discretizes the model defined by the parameters 
    of the continuous-time model a, b, c and d. The method is referred to 
    Euler discretization 

    Args:
        a ([float]): \theta_{0001} (numerator parameter)
        b ([float]): \theta_{0010} (coefficient of s**2 in denominator)
        c ([float]): \theta_{0100} (coefficient of s**1 in denominator)
        d ([float]): \theta_{1000} (coefficient of s**0 in denominator)
        Ts ([float]): Sampling time (in secs)

    Returns:
        model_euler ([state-space]): The state space model of the discretized model
    """
    # simulation with euler discretization
    A = np.array([[0,1,0],
                  [0,0,1],
                  [d,c,b]])

    B = np.array([[0], [0], [a]])
    C = np.array([1, 0, 0])
    #print(A, B, C)
    
    A_euler = np.eye(3) + Ts*A
    B_euler = Ts*B
    C_euler = C

    # construct Euler discretized state space 
    model_euler = ss(A_euler, B_euler, C_euler, 0, 1)
    return model_euler

def plot_all_model_responses(t_arr, y, y_model_CT, y_model_reparam_CT, y_model_euler,
                            y_model_c2d):
    
    plt.figure()
    #plt.figure(figsize=(20,10))
    #plt.rcParams.update({'font.size': 22})
    #plt.subplot(2,1,1)
    plt.plot(t_arr, y,'b', label='measured output')
    #plt.legend(['measured output'])
    #plt.subplot(2,1,2)
    plt.plot(t_arr, y_model_CT, 'c', label='model (CT)', linewidth=2)
    plt.plot(t_arr, y_model_reparam_CT, 'r', label='model reparam. (CT)')
    plt.plot(t_arr, y_model_euler, 'g*-', label='model (Euler)')
    plt.plot(t_arr, y_model_c2d, 'm--', label='model (c2d, zoh)')
    plt.legend()
    plt.show()
    return None

def get_true_params(a, b, c, d, Ts, u, y, t_arr, plot_curves=True):

    Gs = compute_model_tf(a, b, c, d)
    y_model_CT = simulate_model(Gs, u, t_arr*Ts) # Output of CT LTI system
    alpha, poles = get_poles(b, c, d)
    k, alpha, omega0, xi = compute_actual_parameters(a, b, c, d, alpha)
    Hs = compute_actual_tf(k, alpha, omega0, xi)
    y_model_reparam_CT = simulate_model(Hs, u, t_arr*Ts) # Output of reparameterized CT LTI system
    model_euler= euler_discretize_model(a, b, c, d, Ts) # Discretized version of CT model
    y_model_euler = simulate_model(Gs=model_euler, u=u.reshape((-1,))) # Output of Discretized model
    model_zoh_c2d= c2d(sysc=Hs, Ts=Ts, method='zoh') # Discretized version of CT model
    _ = isstable(np.array(ct.pole(model_zoh_c2d)), sys_type='discrete')
    y_model_zoh_c2d = simulate_model(Gs=model_zoh_c2d, u=u.reshape((-1,))) # Output of Discretized model
    if plot_curves== True:
        #simulate_bode(Gs) # Bode plot for the model CT
        plot_all_model_responses(t_arr=t_arr, y=y, 
                                 y_model_CT=y_model_CT, 
                                 y_model_reparam_CT=y_model_reparam_CT, 
                                 y_model_euler=y_model_euler,
                                 y_model_c2d=y_model_zoh_c2d)
        #simulate_bode(ct.matlab.ss2tf(model_euler))
    print("\nK:{:.4f}, alpha:{:.4f}, omega0:{:.4f}, xi:{:.4f}".format(k, alpha, omega0, xi))
    return k, alpha, omega0, xi, y_model_euler

def get_mean_params(list_of_dataset_dicts, Ts, t_arr, input_signal, output_signal, num_parameters=4, plot_curves_flag=False):

    param_array_dataset = np.zeros((len(list_of_dataset_dicts), num_parameters))

    for i in range(len(list_of_dataset_dicts)):
        print("Sample:{}".format(i+1))
        k_i, alpha_i, omega0_i, xi_i, yi_model_euler = get_true_params(**list_of_dataset_dicts[i], 
                                                                    Ts = Ts,
                                                                    u=input_signal, y=output_signal, 
                                                                    t_arr=t_arr, 
                                                                    plot_curves=plot_curves_flag) # Dataset-4
        
        param_array_dataset[i, :] = np.array([k_i, alpha_i, omega0_i, xi_i])
        print("-"*100)
        
    mean_param_array_dataset = np.mean(param_array_dataset[:-1,:], axis=0)
    std_param_array_dataset = np.std(param_array_dataset[:-1,:], axis=0)
    print("-"*100)
    print("Mean values of parameters (except last one):\n{}".format(mean_param_array_dataset))
    print("Std.deviation values of parameters (except last one):\n{}".format(std_param_array_dataset))
    print("-"*100)
    return mean_param_array_dataset, std_param_array_dataset

def get_mean_std_params_prbs(dataset_path_prbs):
    
    print("-"*100)
    print("Extracting mean parameters and std.devs for PRBS dataset")
    print("-"*100)
    z1, u1, z2, u2, z3, u3, N, Ts, model_params_rsys_prbs = get_prbs_dataset(dataset_path=dataset_path_prbs)
    mean_param_array_prbs, std_param_array_prbs = get_mean_params(list_of_dataset_dicts=model_params_rsys_prbs, 
                                                                        Ts = Ts,
                                                                        t_arr=np.linspace(1,N,N), 
                                                                        input_signal=u1, 
                                                                        output_signal=z1, 
                                                                        num_parameters=4, 
                                                                        plot_curves_flag=False)
    return mean_param_array_prbs, std_param_array_prbs

def get_mean_std_params_uniform(dataset_path_uniform):

    print("-"*100)
    print("Extracting mean parameters and std.devs for uniform dataset")
    print("-"*100)
    z11, u11, z12, u12, N, Ts, model_params_rsys_uniform = get_uniform_dataset(dataset_path=dataset_path_uniform)
    mean_param_array_prbs, std_param_array_prbs = get_mean_params(list_of_dataset_dicts=model_params_rsys_uniform, 
                                                                        Ts = Ts,
                                                                        t_arr=np.linspace(1,N,N), 
                                                                        input_signal=u11, 
                                                                        output_signal=z11, 
                                                                        num_parameters=4, 
                                                                        plot_curves_flag=False)
    return mean_param_array_prbs, std_param_array_prbs

def initialize_p0(params_estimate, percent_=0.2):
    # Percentage within mean (+ or - percent_ %) parameter
    eps = np.finfo(float).eps # Get the machine epsilon 
    theta_1 = generate_uniform(N=1, a=params_estimate[0] * (1-percent_), b=params_estimate[0] * (1+percent_)) # Parameter k
    theta_2 = generate_uniform(N=1, a=params_estimate[1] * (1-percent_), b=params_estimate[1]  * (1+percent_)) # Parameter \alpha
    theta_3 = generate_uniform(N=1, a=params_estimate[2] * (1-percent_), b=params_estimate[2]  * (1+percent_)) # Parameter \omega0
    theta_4 = generate_uniform(N=1, a=params_estimate[3] * (1-percent_), b=params_estimate[3]  * (1+percent_)) # Parameter \xi
    #theta_5 = generate_uniform(N=1, a=eps, b=0.01) # Parameter wn_variance (assuming noise is zero-mean)

    theta_vector = [theta_1,
                    theta_2,
                    theta_3,
                    theta_4]

    return theta_vector

def simulate_model_aliter(x, a, b, c, d):
    tf = sig.TransferFunction([a], [1, -b, -c, -d])
    N = len(x)
    t_arr = np.linspace(1, N, N)*20e-3
    to, yo, xo = sig.lsim2(tf, U=x, T=t_arr)
    yo = np.abs(yo)
    return yo
    
def identify_model(t, input_signal, output_signal, method='lm', p0=None, bounds=None):
    
    kwargs = {"epsfcn":1e-9}
    if not p0 is None:
        params, params_cov = curve_fit(simulate_model_aliter, xdata=input_signal, ydata=output_signal,
                                       method=method, p0=p0, **kwargs)
    else:
        params, params_cov = curve_fit(simulate_model_aliter, xdata=input_signal, ydata=output_signal,
                                        method=method)

    return {'a': params[0], 'b': params[1], 'c': params[2], 'd': params[3]}, params_cov

def get_actual_params(a, b, c, d):
    k = -a/d
    alpha = -get_poles(b, c, d)[0]
    omega0 = np.sqrt(-d/alpha)
    xi = (-b - alpha)/(2*omega0)
    return k, alpha, omega0, xi

def get_optimized_params(t_arr, u1, z1, params_estimate):
    
    p0 = initialize_p0(params_estimate=params_estimate, percent_=0.0)
    #param_bounds = ([1.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf])
    params_opt, params_cov_opt =  identify_model(t=t_arr, input_signal=u1.reshape((-1,)), 
                                                 output_signal=z1.reshape((-1,)), method='lm', p0=p0)

    k_opt, alpha_opt, omega0_opt, xi_opt = get_actual_params(**params_opt)

    print("Obtained optimized params: (k, alpha, omega, xi)")
    print([k_opt, alpha_opt, omega0_opt, xi_opt])
    print("Variances for the parameters:")
    print(np.sqrt(np.diag(params_cov_opt)))
    return [k_opt, alpha_opt, omega0_opt, xi_opt], params_opt

def save_params(mean_param_array, std_param_array, dataset_type="prbs", outfile_path=None):

    params_mean_std = {}
    params_mean_std["mean"] = {"k":mean_param_array[0],
                            "alpha": mean_param_array[1],
                            "omega0": mean_param_array[2],
                            "xi": mean_param_array[3]
    }

    params_mean_std["std"] = {"k":std_param_array[0],
                            "alpha": std_param_array[1],
                            "omega0": std_param_array[2],
                            "xi": std_param_array[3]
    }

    if os.path.isfile(os.path.join(outfile_path, "{}_dataset_opt.json".format(dataset_type))) == True:
        # If file already exists, NOTE: Earlier this file was called '[prbs/uniform]_dataset.json'
        print("File already exists!")
        pass
    else:
        # If the file doesn't exist, we create the file and write down the parameters
        with open(os.path.join(outfile_path, "{}_dataset_opt.json".format(dataset_type)), "w") as f: 
            f.write(json.dumps(params_mean_std, cls=NDArrayEncoder, indent=2))

def main():

    #Ts = 20e-3
    #N = 500
    dataset_path_prbs = "../Real-World-Experiments/CoupledElectricDrives/CoupledElectricDrivesDataSetAndReferenceModels/DATAPRBS.MAT"
    dataset_path_uniform = "../Real-World-Experiments/CoupledElectricDrives/CoupledElectricDrivesDataSetAndReferenceModels/DATAUNIF.MAT"
    
    mean_param_array_prbs, std_param_array_prbs = get_mean_std_params_prbs(dataset_path_prbs)
    mean_param_array_uniform, std_param_array_uniform = get_mean_std_params_uniform(dataset_path_uniform)                                                                   
    
    ##################################################################################################
    # Getting a set of optimized parameters for the PRBS dataset by directly optimizing using 
    # available input-output data
    ##################################################################################################
    print("------- Calculating and saving optimized parameters --------")
    z1, u1, _, _, _, _, N, Ts, model_params_rsys_prbs = get_prbs_dataset(dataset_path=dataset_path_prbs)
    t_arr = np.linspace(1, N, N)*Ts

    params_mean_estimate = np.zeros((4,))

    for i in range(len(model_params_rsys_prbs)-1):
        params_mean_estimate += np.array(list(model_params_rsys_prbs[i].values()))
    
    params_mean_estimate = params_mean_estimate / (len(model_params_rsys_prbs) - 1)
    print("Mean estimates of [a, b, c, d]: \n{}".format(params_mean_estimate))

    tf_params_opt, params_opt = get_optimized_params(t_arr, u1, z1, params_estimate=params_mean_estimate)
    
    params_opt_dict = {}
    params_opt_dict["mean"] = {"k":tf_params_opt[0],
                                "alpha": tf_params_opt[1],
                                "omega0": tf_params_opt[2],
                                "xi": tf_params_opt[3]
                            }

    print(tf_params_opt)
    print(params_opt_dict)

    Hs_opt = sig.TransferFunction(params_opt["a"], [1, -params_opt["b"], -params_opt["c"], -params_opt["d"]])
    y_model_brute_force_optim = np.abs(sig.lsim2(Hs_opt, u1.reshape((-1, 1)), t_arr)[1])

    plt.figure(figsize=(15,5))
    #plt.subplot(2,1,1)
    plt.plot(t_arr, z1,'b')
    #plt.legend(['measured output'])
    #plt.subplot(2,1,2)
    plt.plot(t_arr, simulate_model_aliter(u1, **params_opt), 'r')
    plt.legend(['measured output', 'model (CT, Brute force optim)'])
    #plt.savefig('./brute_force_optimization.pdf')
    plt.show()
    
    MSE_after_optimization = scipy.linalg.norm(z1.reshape((-1, 1)) - y_model_brute_force_optim.reshape((-1, 1)))**2

    print("MSE values:")
    print("After brute force optimization: {}".format(MSE_after_optimization))

    with open(os.path.join("./data/coupled_drive", "prbs_dataset_opt.json"), "w") as f: 
        f.write(json.dumps(params_opt_dict, cls=NDArrayEncoder, indent=2))

    #save_params(mean_param_array_prbs, std_param_array_prbs, dataset_type="prbs", outfile_path="./data/coupled_drive/")
    #save_params(mean_param_array_uniform, std_param_array_uniform, dataset_type="uniform", outfile_path="./data/coupled_drive/")
    
    return None

if __name__ == "__main__":
    main()