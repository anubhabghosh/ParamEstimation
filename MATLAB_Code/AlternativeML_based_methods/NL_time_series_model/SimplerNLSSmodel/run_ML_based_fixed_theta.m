clear;
clc;
close all;

%==================================================
%===   Load the model from the Python data   ===
%==================================================
ZN = load('evaluation_ZN_100_fixed_samples_simpler_alt_model.mat');
num_trajs = ZN.num_trajectories;
num_realizations = ZN.num_realizations;
ZN_lens = ZN.trajectory_lengths;
N = mean(ZN_lens);
mse_predicted_actual_theta = 0;
M = cast(num_trajs*num_realizations, 'int64');
mse_vector = zeros(M, 1);
t_elapsed_vec = zeros(M, 1);
% Start a timer at the beginning of the loop to measure the time elapsed in
% total
tic;
minTimeElapsed = inf;

%==================================================
% Process starts 
%==================================================
for mi=1%:M
    
    disp('%====================================================%');
    disp(['Sample index : ', num2str(mi)]);
    
    y_mi = ZN.data{mi, 2}; % Get the data trajectory
    sampled_theta_mi = ZN.data{mi, 1}; % Get the theta vectors
    
    tstart = tic; % Timer start
    
    % Get the prediction
    predicted_theta_mi = evaluate_using_NLSS_fixed_theta_simpler_model(transpose(y_mi), ...
                                                                   sampled_theta_mi, N); 
    
                                               telapsed = toc(tstart);
    minTimeElapsed = min(minTimeElapsed, telapsed);
    t_elapsed_vec(mi, :) = minTimeElapsed;
    disp(['Time elapsed (in secs) for current sample: ', num2str(minTimeElapsed)]);
    
    % Sum the mean squared error predictions over the total number of
    % samples M
    mse_predicted_actual_theta = mse_predicted_actual_theta + ...
                         immse(transpose(sampled_theta_mi), predicted_theta_mi);
    
    % Store the MSE predictions on a per sample basis
    mse_vector(mi,:) = immse(transpose(sampled_theta_mi), predicted_theta_mi);
    disp(['MSE for current sample: ', num2str(mse_vector(mi,:))]);
    disp('%====================================================%');
    
end

%==================================================
% Display the final result
%==================================================
avg_mse = mse_predicted_actual_theta / double(M);
disp(['Estimated MSE using ML based method: ', num2str(avg_mse)]);
disp(['Average time elapsed in secs (per sample) for total process: ', num2str(toc/M)]);

%==================================================
% Saving the results
%==================================================
res.avgmse = avg_mse;
res.msevec = mse_vector;
res.t_elapsed_vec = t_elapsed_vec;
save('results_ML_fixed_theta_simpler_altmodel.mat', 'res');
