# ParamEstimation
Designing Recurrent neural networks for parameter estimation of non-linear models

## Estimating parameters of a Non linear model:

Characteristics of the Non linear model:
- Parameters: `theta_vector = [0.5, 25, 1, 8, 0.05, 1, 0.1]`,
  - Possibly left out: `theta_3=1.0` (fixed during data generation),
  - Possibly left out: `theta_4=8.0` (not globally identifiable). You ask the model to predict the value, but don't include in the error calculation in the evaluation stage / testing stage. 
  
## Tasks ([x] - Completed, [ ] - Pending) :
- [x] Train an RNN model to predict the parameters of the model (trying to predict all 7 parameters) (*NOTE: base result needs improvement*)
- [x] Grid search to find optimal RNN parameters based on validation loss. Obtained a good GRU model, with tuned set of parameters.
- [x] Train an RNN model to predict the last two parameters `\theta_6, \theta_7`.
- [x] Train an RNN model to predict 5 specific parameters (2 other parameters fixed!).
- [x] Comparison with Expectation-Maximization based Particle Smoother method (EMPS method).
- [x] Minor experiment: Robustness study - Examining the effect of adding noise to driving noise in order to simulate mismatched training-testing conditions. This means that the trajectories in test time will contain some minor noise (such as a phase shift in driving noise equation), that was unseen during training. 
- [ ] Perform extensive simulation for `pfixed` mode using RNN models with varying size of dataset. We keep initially `N=200`, and vary `M` and `P` over a grid of values. 
  - [x] Refactor code to automate process of dataset creation in respetcive folders. Folders are to be named after the dataset configuration present run.
  - [ ] Refactor code to automate process of running RNN models and creating separate logfiles in appropriate folders
  - [ ] Refactor code to automate process of storing only two model checkpoints: *best_model* and *last_model* for each combination of parameters (of the dataset) in separate model folders
- [ ] Simulate trajectories for a real-world dataset such as [coupled electric drives](https://sites.google.com/view/nonlinear-benchmark/benchmarks/coupled-electric-drives).
  - [x] Generate the parameters for the Continuous time transfer function `(k, alpha, xi, omega0)`, from the ones reported in the [paper](http://www.it.uu.se/research/publications/reports/2017-024/2017-024-nc.pdf). Take care of the signs of the parameters obtained.
  - [ ] Choose *a reasonable prior* support for the parameters: 
    - [ ] A first experiment could for example be to sample within 10% to 20% around the best value from the coupled drive paper. *Future work:* The upper and lower limits of the support intervals can be seen as **hyper-parameters** and ideally can be tuned using a method like **cross-validation**. 
    - [ ] Use domain knowledge to choose reasonable limits for the priors. For e.g. `omega0` (natural frequency in rad/s) should be positive, `xi` should be between -1 and 0, `alpha` should be negative as it is a pole, `k` shouldn't be arbritrarily too high. Even additive noises should be zero-mean with small variances ~ 0.01.
  - [x] Formulate the transfer function using the above set of parameters `(k, alpha, xi, omega0)`, and perform **exact discretization** (tuning parameters for discrete case **actually** results in tuning the same for continuous case). 
  - [ ] Accuracy estimation: Visual inspection by simulating tajectories using `lsim` or computing some form of measure between two simulated trajectories.
  

### Estimating variances `theta_6, theta_7` (mode - `vars`): 
We tried to simulate the model for just estimating the variances. The parameters used for generating the training data were: `M=500, P=50`, where `M` denotes the number of trajectories and `P` denotes the number of parameter vectors drawn. Also, as a variation in the architecture, an extra hidden layer was added after the recurrent network layers, so that a better estimate could be obtained. The sequence length / trajectory length `N = 200`.

### Estimating the vector `theta_1, theta_2, theta_5, theta_6, theta_7` (mode - `pfixed`):
We fix the parameter values of `theta_3, theta_4` to be `1.0` and `8.0` respectively. We also ran experiments with the modified training data set taking values of theta vector from your specified range, and using for `M`, `P` i.e. `M=500, P=50, N=200`. After training the models on the data set, they were evaluated on an evaluation set. We sampled `M=100` values of `[theta_1, theta_2, theta_5, theta_6, theta_7]` and for each sample, generated one trajectory for each sampled theta. This is different from the setup for evaluation described in subsection, where for the evaluation data set we generate multiple trajectories from a single, fixed *true* value of `theta_vector = 0.5, 25, 1, 8, 0.05, 1, 0.1`. As opposed to this, in this experiment, we sample values of `theta_vector` as per limits defined for priors, and generate one trajectory for each realization of  `theta_vector`.

### Performing experiments on real-world data:
Experiments performed using datasets from [this paper](http://www.it.uu.se/research/publications/reports/2017-024/2017-024-nc.pdf). There are datasets corresponding to two input signals: 
  - Uniform signal (DATAUNIF.MAT)
  - PRBS signal (DATAPRBS.MAT)

For now, we focus on only experimenting with PRBS dataset.
#### *NOTE:* 
- When generating the training data for the coupled drive example, we should use either the `lsim` function directly on the continuous-time linear model or the `c2d` function with a `'zoh'` option, then simulate the returned linear discrete-time model, then finally take the absolute value to get the output trajectory. This way we are doing the simulation exactly with no approximation error (under the `'zoh'` assumption on the input). *NOTE:* We are **not** concerned with the Euler discretization here; this was used in the [paper](http://www.it.uu.se/research/publications/reports/2017-024/2017-024-nc.pdf) because the algorithm they used was designed to work on general Non-linear models (*not necessarily linear dynamical model followed by a static NL function*), so the algorithm is designed to discretize a general nonlinear model, and to do that they used the approximate method of Euler.
-  During training, there is no real data. Most of it is synthetic, and can be optimized using MSE as earlier. If we plot curves similar to MATLAB file sent by *Mohamed*, one of these curves is the real data trajectory, the second is the simulated output using exact discretization (this is what we should look at), and the last curve is the simulated output using Euler’s method. You will notice that the Euler’s curve is closer to the real data compared to the exact discretization curve. The reason for this is that the algorithm they used fits an Euler discretized model and not an exactly discretized model, which is a disadvantage of [their method in section 3.5](http://www.it.uu.se/research/publications/reports/2017-024/2017-024-nc.pdf). We should fit an exactly discretized model, so only concerned with trying to make the simulated trajectory using the CT-model (learned parametes come from trained model) to be as close to the measured signal as we can. 
