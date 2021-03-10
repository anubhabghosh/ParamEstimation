# ParamEstimation
Designing Recurrent neural networks for parameter estimation of non-linear models

## Estimating parameters of a Non linear model:

Characteristics of the Non linear model:
- Parameters: $\theta_{initial} = \[0.5, 25, 1, 8, 0.05, 1, 0.1 \]$
  - Possibly left out: $\theta_{3}=1.0$ (fixed during data generation),
  - Possibly left out: $\theta_{4}=8.0$ (not globally identifiable). You ask the model to predict the value, but don't include in the error calculation in the evaluation stage / testing stage. 
  
### Completed:
- [x] Train an RNN model to predict the parameters of the model (trying to predict all 7 parameters) (*NOTE: base result needs improvement*)
- [x] Grid search to find optimal RNN parameters based on validation loss. Obtained a good GRU model, with tuned set of parameters.
### TODO:
- [ ] **For MATLAB simulations**: 
    - It is important to ensure that the initial values for Maximum likelihood based estimation are not too far from the true value, as some noise is initially added. If the value is too far, then the optimization process might result in a local minimum. 
    - True values should be set properly during the estimation.
- [ ] **Trian an RNN model to predict only the variance parameters $\theta_{6},\theta_{7}$, whose actual values are 1.0 and 0.1 respectively**.
    - [ ] Generate new data fixing other parameters and genrating trajectories based on realizations of $\theta_{6},\theta_{7}$.
    - [ ] Compare the estimate with that computed using the `cpf.m` MATLAB code. Important thing to consider is that the dataset generation process is the same in Python and MATLAB (same parameters as well as starting indices such as starting with u(k) as opposed to u(k+1) etc.)
    - [ ] Check if the prediction could be improved by a weighting process. Weight the error between true parameter $\theta$ and estimated parameter $\Theta_{\alpha}$ (L2 loss / MSE loss) by a weight vector $\mathbf{w}$ s.t. $\mathbf{w} = \[ \frac{1}{mean(range of \theta_{6})}, \frac{1}{mean(range of \theta_{7})}\]$. This weight can later be updated as $\mathbf{w} = \[ \frac{1}{\text{calculated estimate of }\theta_{6}}, \frac{1}{\text{calculated estimate of } \theta_{7}}\]$.
    - [ ] Grid search may be required as model is now changed, because we fix first parameters as the true values and train to predict only the final two parameters. For now, can try with the obtained grid-search based parameters.
- [ ] **Trian an RNN model to predict only 6 parameters except $\theta_{3}**.
    - [ ] Compare the estimate with that computed using the `EMallNLSS.m` MATLAB code. Important thing to consider is that the dataset generation process is the same in Python and MATLAB (same parameters as well as starting indices such as starting with u(k) as opposed to u(k+1) etc.)
    - [ ] Check if the prediction could be improved by a weighting process. Weight the error between true parameter $\theta$ and estimated parameter $\Theta_{\alpha}$ (L2 loss / MSE loss) by a weight vector $\mathbf{w}$ s.t. $\mathbf{w} = \[ \frac{1}{mean(range of \theta_{1})}, \frac{1}{mean(range of \theta_{2})}, ..., \frac{1}{mean(range of \theta_{7})}\]$. This weight can later be updated as $\mathbf{w} = \[ \frac{1}{\text{calculated estimate of }\theta_{1}}, \frac{1}{\text{calculated estimate of } \theta_{2}}, ..., \frac{1}{\text{calculated estimate of } \theta_{7}}\]$.
    - [ ] Grid search may be required as model is now changed, because we fix first parameters as the true values and train to predict only 6 parameters.
- [ ] **Train an RNN model to predict the parameters of the model (all 6/7 parameters) [Main objective]**
