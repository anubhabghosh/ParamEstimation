
function [theta_samples] = compute_cond_mean_pMCMC(y, theta_actual, M, T_burn_in, T_reqd)

%====================================================
%==================  True Model  ====================
%====================================================
a0 = theta_actual(1);
b0 = theta_actual(2);
Q0 = theta_actual(3);
R0 = theta_actual(4);

theta0 = [a0 b0 sqrt(Q0) sqrt(R0)];  % true parameter
z.y = y;  % data set
u = cos(1.2.*linspace(1, length(y), length(y)));  % Cosine input
z.u = u;
% %==================================================
% %===   pMCMC via auxilary PF and MH algorithm   ===
% %==================================================
%=====================================================================
alpha = 0.4; % Tunable parameter for MH algorithm
% alpha_arr = [0.01];% 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6];
% T = T_burnin + T_required % Only start considering after burn-in number
% of samples
% Thinning: Skip l number of samples, post the burn-in samples, so this
% requires more simulations, to improve the variance
%T_burn_in = 10^3;
%T_reqd = 2500;
T = T_burn_in + T_reqd;

% Number of samples to skip post burn in samples to store the states of 
% the Markov chain
% thinning_interval = 10; 

%=====================================================================
%formatspec = "%s theta vector: a: %.4f, b: %.4f, c: %.4f, d: %.4f, Q: %.4f, R: %.4f \n"; 
%disp('------------------------------------------------------------------------------------');
%disp(['Alpha: ', num2str(alpha)]);
disp(['Number of MH iterations: ', num2str(T)]);
% returns a markov chain, now using the new reparameterized computation
theta_samples = pMCMC_new(z, M, theta0, alpha, T);  
