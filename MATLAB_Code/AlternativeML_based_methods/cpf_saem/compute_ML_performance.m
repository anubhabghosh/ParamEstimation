%%
%==================================================
%===   Load the model from the Python data   ===
%==================================================

clear;
clc;
close all;

DN = load('evaluation_ZN_100_fixed_samples_latest.mat');
num_realizations = DN.num_realizations; % The number of sample realizations (P)
DN_lens = DN.trajectory_lengths; % The trajectory lengths
N = mean(DN_lens); % It is assumed that every trajectory has equal lengths, i.e. N=200
num_trajs = DN.num_trajectories; % The number of trajectories (M) for every sampled parameter (p)
mse_predicted_actual_theta = 0; 
M = cast(num_trajs*num_realizations, 'int64'); % Getting the total number of sample points (P * M)
mse_vector = zeros(M, 1);
list_of_predictions = zeros(M,2);

for mi=1:M
    
    %if mod(m_i, 20) == 0
    disp(['Sample index : ', num2str(mi)]);
    %end
    y_mi = DN.data{mi, 2}; % Get the data
    
    actual_theta_mi = DN.data{mi, 1}; % Get the vectors
    
    % Get the prediction
    [q1, r1] = evaluate_using_cpf(transpose(y_mi), N, M); 
    
    predicted_theta_mi = [q1(end) r1(end)]; % take the values at the end of 500 or so iterations of EM
    %disp(size(predicted_theta_mi))
    %disp(size(actual_theta_mi))

    % Sum the mean squared error predictions
    mse_predicted_actual_theta = mse_predicted_actual_theta + immse(transpose(actual_theta_mi), predicted_theta_mi);
    
    mse_vector(mi,:) = immse(transpose(actual_theta_mi), predicted_theta_mi);
    fprintf("MSE calculated for sample index %d is %f \n", mi, mse_vector(mi, :)); 
    list_of_predictions(mi, :) = predicted_theta_mi;
end
    
avg_mse = mse_predicted_actual_theta / double(M);
res.avgmse = avg_mse;
res.msevec = mse_vector;
res.list_of_predictions = list_of_predictions;
filename = sprintf('results_ML_ZN_%d_N%d_CPF_latest.mat',M,N);
save(filename, 'res');

disp(['Estimated MSE using ML based method: ', num2str(avg_mse)]);
