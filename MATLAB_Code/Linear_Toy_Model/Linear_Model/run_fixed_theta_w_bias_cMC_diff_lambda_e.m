clc;
clear;
close all;

p=1000;
M = 500;
K_test = 1000;
d = 2;
%N = 200;
N=500; % Nominally the number of samples
%theta_i = 2*rand(d, p); % Uniform prior
var_theta = [1/3 0; 0 1/3;];
mu_theta = [1; 1;];
theta_i = 1.0 + chol(var_theta)*randn(d, p); % Gaussian prior

evaluation_results_trial = load('Results_LM_Fixed_theta_N500_GaussPrior_plus_num_bias_trial_4.mat');

A_analytical_trial = evaluation_results_trial.evaluation_results.stats(end).A_analytical;
b_analytical_trial = evaluation_results_trial.evaluation_results.stats(end).b_analytical;
%A_asymptotic_trial = evaluation_results_trial.evaluation_results.stats(16).A_asymptotic;
%b_asymptotic_trial = evaluation_results_trial.evaluation_results.stats(16).b_asymptotic;
Phi = evaluation_results_trial.evaluation_results.Phi;

n_MC = 100;
actual_theta = 0.7.*ones(K_test, d, 1);
lambda_e_arr = reshape(0.1:0.1:5.0,[50, 1]); %reshape(linspace(0.1, 1.0, 21).^2, [21, 1]);
MSE_stats_MC = zeros(length(lambda_e_arr), n_MC);
for trial=1:n_MC
    
    theta_stats_analytical = zeros(K_test, d, length(lambda_e_arr));
    theta_stats_asymptotic = zeros(K_test, d, length(lambda_e_arr));
   
    %U = randn(N,1);
    %Phi = [U [0; U(1:N-1);]];
    for k = 1:length(lambda_e_arr)
        lambda_e = lambda_e_arr(k);
        %disp(['Creating evaluation data for estimation...', num2str(lambda_e)]);
        Y = zeros(K_test, N);
        for i=1:size(actual_theta, 1)
            E = sqrt(lambda_e).*randn(N,1);
            Y(i, :) = Phi*actual_theta(i, :)' + E;
        end
        
        [theta_hat_asymptotic, A_asymptotic_trial, b_asymptotic_trial] = compute_asymptotic_theta_estimate(d, p, N, theta_i, Phi, Y, mu_theta, var_theta, lambda_e);
        theta_hat_analytical = A_analytical_trial *Y' + b_analytical_trial;
        %theta_hat_asymptotic = A_asymptotic_trial *Y' + b_asymptotic_trial;
        %theta_hat_analytical_2 = A_analytical_trial_2*Y' + b_analytical_trial_2;
        theta_stats_analytical(:, :, k) = theta_hat_analytical';
        theta_stats_asymptotic(:, :, k) = theta_hat_asymptotic;
        %theta_stats_2(:, :, k) = theta_hat_analytical_2';
        
    end
    MSE_stats = reshape(mean((theta_stats_asymptotic - theta_stats_analytical).^2, [1,2]), [length(lambda_e_arr), 1]);
    MSE_stats_MC(:, trial) = MSE_stats;
end
%% Display visual result
MSE_stats_MC_log = log10(MSE_stats_MC);
figure;
plot(lambda_e_arr', log10(mean(MSE_stats_MC, 2)' - std(MSE_stats_MC, 1, 2)'), 'r', 'LineWidth', 1.0);
hold on;
plot(lambda_e_arr', log10(mean(MSE_stats_MC, 2)' + std(MSE_stats_MC, 1, 2)'), 'b', 'LineWidth', 1.0);
hold on;
patch([lambda_e_arr' fliplr(lambda_e_arr')], [log10(mean(MSE_stats_MC, 2)' - std(MSE_stats_MC, 1, 2)') fliplr(log10(mean(MSE_stats_MC, 2)' + std(MSE_stats_MC, 1, 2)'))], 'g');alpha(0.3);
hold on;
plot(lambda_e_arr, mean(MSE_stats_MC_log, 2), 'b-', 'LineWidth', 2, 'Marker', 's');

figure;
plot(lambda_e_arr', mean(MSE_stats_MC, 2)' - std(MSE_stats_MC, 1, 2)', 'r', 'LineWidth', 1.0);
hold on;
plot(lambda_e_arr', mean(MSE_stats_MC, 2)' + std(MSE_stats_MC, 1, 2)', 'b', 'LineWidth', 1.0);
hold on;
patch([lambda_e_arr' fliplr(lambda_e_arr')], [mean(MSE_stats_MC, 2)' - std(MSE_stats_MC, 1, 2)' fliplr((mean(MSE_stats_MC, 2)' + std(MSE_stats_MC, 1, 2)'))], 'g');alpha(0.3);
hold on;
plot(lambda_e_arr, mean(MSE_stats_MC, 2), 'b-', 'LineWidth', 2, 'Marker', 's');
