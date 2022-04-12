
function [theta_samples] = compute_cond_mean_pMCMC(y, u, theta_actual, prbs_struct, T_burn_in, T_reqd, Ts)
%====================================================
%==================  True Model  ====================
%====================================================
k_0 = theta_actual(1);
alpha_0 = theta_actual(2);
w0_0 = theta_actual(3);
xi_0 = theta_actual(4);
e_var_0 = theta_actual(5);

theta0 = [k_0 alpha_0 w0_0 xi_0 sqrt(e_var_0)];  % true parameter
z.y = y;  % data set
z.u = u;

%=========================================================================%
%=================  Set some parameters for pMCMC  =======================%
%=========================================================================%

%alpha = 0.00005; % Tunable parameter controlling variance for MH algorithm
alpha = 3e-5;
percent_ = 0.2; % Percentage to choose prior from
T = T_burn_in + T_reqd; % Only start considering after burn-in number
% of samples

%=========================================================================%
%================= Thinning interval (kept 'OFF' for now) ================%
% Thinning: Skip l number of samples, post the burn-in samples, so this
% requires more simulations, to improve the variance
% thinning_interval = 10; 

%==============% returns a markov chain, now using the new reparameterized 
% computation ===============
% formatspec = "%s theta vector: a: %.4f, b: %.4f, c: %.4f, d: %.4f, 
% Q: %.4f, R: %.4f \n"; 
%disp('-----------------------------------------------------------------');
%disp(['Alpha: ', num2str(alpha)]);
theta_samples = pMCMC_ce_drive(z, theta0, prbs_struct, alpha, T, percent_, Ts);
end
