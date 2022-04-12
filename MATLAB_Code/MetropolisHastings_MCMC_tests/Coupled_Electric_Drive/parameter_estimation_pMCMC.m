clear;
clc;
close all;
%% load data
D = load('DATAPRBS.mat'); % Enter the full name of the datafile
Ts = 20e-3; % Sample time
z1 = D.z1; % Output signal
u1 = D.u1; % Input signal
%% %======================================================================%
%===   pMCMC via auxilary PF and MH algorithm   ===%
%=========================================================================%
%alpha_arr = [0.01];% 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6];
% T = T_burnin + T_required % Only start considering after burn-in number
% of samples
% Thinning: Skip l number of samples, post the burn-in samples, so this
% requires more simulations, to improve the variance
T_burnin = 2.0e3;
T_required = 3.5e3;

% Number of samples to skip post burn in samples to store the states of 
% the Markov chain
thinning_interval = 10; 
    
%=====================================================================
disp('------------------------------------------------------------------------------------');
%disp(['Alpha: ', num2str(alpha)]);
disp(['Number of MH iterations: ', num2str(T_burnin + T_required)]);

% Load the 'brute-force' optimized parameters
prbs_struct = load('prbs_dataset_opt.mat');

% returns a markov chain, now using the new reparameterized computation
k_true = prbs_struct.prbs_dataset_bf_opt.k_bf;
alpha_true = prbs_struct.prbs_dataset_bf_opt.alpha_bf;
w0_true = prbs_struct.prbs_dataset_bf_opt.omega0_bf;
xi_true = prbs_struct.prbs_dataset_bf_opt.xi_bf;
e_var_true = 1e-6; % Assuming very small noise variance

theta0 = [k_true alpha_true w0_true xi_true e_var_true];

theta_samples = compute_cond_mean_pMCMC(z1, u1, theta0, ...
    prbs_struct.prbs_dataset_bf_opt, T_burnin, T_required, Ts);

%% estimate of conditional mean
formatspec = "%s theta vector: k: %.4f, alpha: %.4f, omega0: %.4f, xi: %.4f, e_k_var: %.8f \n";
 % an estimate of E[theta|Y ] = conditional mean, only after samples for the burn-in period
thetaHat = mean(theta_samples(T_burnin:end, :)); 
fprintf(formatspec, "True", theta0(1), theta0(2), theta0(3), theta0(4), theta0(5).^2); 
disp('------------------------------------------------------------------------------------');
fprintf(formatspec, "Estimated", thetaHat(1), thetaHat(2), thetaHat(3), thetaHat(4), thetaHat(5).^2);
%mse = immse([theta0(1), theta0(2), theta0(3), theta0(4), theta0(5).^2], ...
%    [ thetaHat(1), thetaHat(2), thetaHat(3), thetaHat(4), thetaHat(5).^2]);

% Formulate transfer function
k_hat = thetaHat(1);
alpha_hat = thetaHat(2);
w0_hat = thetaHat(3);
xi_hat = thetaHat(4);
e_k_var_hat = thetaHat(5);

[G_discrete_MH, x_model_hat_MH] = get_discr_model(u1, k_hat, alpha_hat, w0_hat, xi_hat, Ts);
y_model_hat_MH = x_model_hat_MH; %+ randn(size(x_model_hat_MH, 1), 1).*sqrt(e_k_var_hat);
disp(["Is constructed transfer function stable?", num2str(isstable(G_discrete_MH))]);

L2_Norm = norm(z1 - y_model_hat_MH, 2).^2;
disp(['Squared L2 norm distance between actual and MH simulated: ', num2str(L2_Norm)]);
disp('------------------------------------------------------------------------------------');



%% Plot the inputs
figure(1);
subplot(2,1,1);
plot((1:500).*Ts, z1);
grid on;
subplot(2,1,2);
plot((1:500).*Ts, u1);
grid on;
%hold all
%% Plot the reslting simulations
figure;
plot((1:500).*Ts, z1);
hold on;
plot((1:500).*Ts, y_model_hat_MH);
hold off;
legend('Measured Output', 'model (MH)');
%hold all
%% Plot the Markov chain results
figure;
subplot(411);
plot(theta_samples(T_burnin:end, 1));title('k');
subplot(412);
plot(theta_samples(T_burnin:end, 2));title('alpha');
subplot(413);
plot(theta_samples(T_burnin:end, 3));title('omega0');
subplot(414);
plot(theta_samples(T_burnin:end, 4));title('xi');
%subplot(515);
%plot(theta_samples(T_burnin:end, 5).^2);title('variance e');
%% Plot the Markov chain results histograms
figure;
subplot(221);
histogram(theta_samples(T_burnin:end, 1));title('k');
subplot(222);
histogram(theta_samples(T_burnin:end, 2));title('alpha');
subplot(223);
histogram(theta_samples(T_burnin:end, 3));title('omega0');
subplot(224);
histogram(theta_samples(T_burnin:end, 4));title('xi');
%% Save results
% res_MC_ce_drive.theta_samples = theta_samples;
% res_MC_ce_drive.G = G_discrete_MH;
% res_MC_ce_drive.theta_hat = thetaHat;
% res_MC_ce_drive.L2_norm =  L2_Norm;
% save('results_CE_Drive_MCMC.mat', 'res_MC_ce_drive');