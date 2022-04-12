clc;
clear;
close all;

p_arr = [1000]; %[400 500 600 1000];
M_arr = [500]; %[50 100 200 500];
K_test = 1000;
d = 2;
%N = 200;
N=500; % Nominally the number of samples

% data set for evaluation
%theta = [1; 1;];
actual_theta = 0.7.*ones(K_test, d, 1);
lambda_e = 1.0^2; %0.3^2;
disp('Creating evaluation data for estimation...');
Y = zeros(K_test, N);
for i=1:size(actual_theta, 1)
    E = sqrt(lambda_e).*randn(N,1);
    U = randn(N,1);
    Phi = [U [0; U(1:N-1);]];
    Y(i, :) = Phi*actual_theta(i, :)' + E;
end
%Y = [Y ones(K_test,1)]; % Adding a '1' to the vector of inputs
num_trials = 3;
count = 1; count = cast(count, 'int64');
MSE_matrix = zeros(length(p_arr), length(M_arr));
%evaluation_results = zeros(length(p_arr)*length(M_arr), 1);
%statistics = zeros(cast(length(p_arr)* length(M_arr), 'int64'), 7);
statistics = struct();
for i=1:length(p_arr)
    for j=1:length(M_arr)
        [mse_pred_actual, mse_pred_asymp, predicted_theta, asymp_theta,  actual_theta, A_analytical, b_analytical, A_asymptotic, b_asymptotic] = compute_test_mse_fixed_theta_w_bias(d, M_arr(j), p_arr(i), N, Phi, Y, actual_theta, lambda_e);
        MSE_matrix(i, j) = mse_pred_asymp;
        disp(['MSE between pred. and asymptotic. for Dataset M:', num2str(M_arr(j)), ' and P:', num2str(p_arr(i)), ' N:', num2str(N), ' is =', num2str(mse_pred_asymp)]);
        disp(['MSE between pred. and actual for Dataset M:', num2str(M_arr(j)), ' and P:', num2str(p_arr(i)), ' N:', num2str(N), ' is =', num2str(mse_pred_actual)]);
        statistics(count).M = M_arr(j);
        statistics(count).P = p_arr(i);
        statistics(count).N = N;
        statistics(count).mse_pred_actual = mse_pred_actual;
        statistics(count).mse_pred_asymp = mse_pred_asymp;
        statistics(count).norm_pred_actual = norm(predicted_theta-actual_theta, 2);
        statistics(count).norm_pred_asymp = norm(predicted_theta-asymp_theta, 2);
        statistics(count).A_analytical = A_analytical;
        statistics(count).b_analytical = b_analytical;
        statistics(count).A_asymptotic = A_asymptotic;
        statistics(count).b_asymptotic = b_asymptotic;
        disp(['L2 norm between actual and pred. parameters is =', num2str(norm(predicted_theta-actual_theta, 2))]);
        disp(['L2 norm between asymp. and pred. parameters is =', num2str(norm(predicted_theta-asymp_theta, 2))]);
        count = count + 1;
        disp("-----------------------------------------------------------------------------------------------------------");
    end
end
%MSE_matrix_trials(:, :, k) = MSE_matrix;
%mean_MSE_matrix = mean(MSE_matrix_trials, 3);
%std_MSE_matrix = sqrt((1.0/num_trials).*sum((MSE_matrix_trials - mean_MSE_matrix).^2, 3));

evaluation_results.stats = statistics;
evaluation_results.Phi = Phi;
evaluation_results.actual_theta = actual_theta;
evaluation_results.lambda_e = lambda_e;
evaluation_results.MSE_matrix = MSE_matrix;
MSE_matrix_table = array2table(MSE_matrix, 'RowNames', {'400', '500', '600', '1000'}, 'VariableNames', {'50', '100', '200', '500'});
evaluation_results.MSE_matrix_table = MSE_matrix_table;
save('Results_LM_Fixed_theta_N500_GaussPrior_plus_num_bias_trial_4.mat', 'evaluation_results');
%% Display visual result
figure; imagesc(table2array(MSE_matrix_table)); colorbar;