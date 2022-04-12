clear;
clc;
close all;

%==================================================
%===   Load the model from the Python data   ===
%==================================================
ZN = load('evaluation_ZN_100_fixed_samples_latest.mat'); 
%ZN = load('evaluation_DN_1000_N_200_samples_var.mat');
num_trajs = ZN.num_trajectories;
num_realizations = ZN.num_realizations;
ZN_lens = ZN.trajectory_lengths;
N = mean(ZN_lens);
mse_predicted_fixed_theta_sum = 0;
%M = cast(num_trajs*num_realizations, 'int64');
M = 100;

% pMCMC parameters
T_burnin = 1e3;
T_reqd = 3.5e3;

num_particles = 10000; % number of particles in the PF

mse_vector = zeros(M, 1);
preds_vector = zeros(M, 2);
preds_vector_arr = zeros(M, T_burnin + T_reqd, 2);
acts_vector = zeros(M, 2);
t_elapsed_vec = zeros(M, 1);

% Start a timer at the beginning of the loop to measure the time elapsed in
% total
tic;
minTimeElapsed = inf;

%==================================================
% Process starts 
%==================================================
% Problematic indices: [3 35 38 48 51 57 70 72 74 77 90 94]
% Coressponding MSE: [0.0484 0.02050 0.0267 0.0484 0.1091 0.0222 0.0247 0.0358 0.0227 0.0815 0.0241 0.0575]

for mi=1:M%[3 35 38 48 51 57 70 72 74 77 90 94]%1:M
    
    
    disp('%====================================================%');
    disp(['Sample index : ', num2str(mi)]);
    
    y_mi = ZN.data{mi, 2}; % Get the data trajectory
    fixed_theta_mi = ZN.data{mi, 1}; % Get the theta vectors
    
    tstart = tic; % Timer start
    disp(['Actual parameter', '  Q: ', num2str(fixed_theta_mi(1)),'  R: ',num2str(fixed_theta_mi(2))]);
    disp('%====================================================%');
    % Get the prediction
    predicted_theta_mi_mh_arr = compute_cond_mean_pMCMC(y_mi, fixed_theta_mi, num_particles, T_burnin, T_reqd);

    telapsed = toc(tstart);
    minTimeElapsed = min(minTimeElapsed, telapsed);
    t_elapsed_vec(mi, :) = minTimeElapsed;
    disp(['Time elapsed (in secs) for current sample: ', num2str(minTimeElapsed)]);
    
   
    
    % estimate of conditional mean: an estimate of E[theta|Y ] = 
    % conditional mean, only after samples for the burn-in period
    predicted_theta_mi = mean(predicted_theta_mi_mh_arr(T_burnin:end, :)); 
    formatspec = "Estimated theta (after %d iterations) : Q: %.4f, R: %.4f,\n";
    fprintf(formatspec, T_burnin + T_reqd, predicted_theta_mi(1).^2, predicted_theta_mi(2).^2);
    
    % Store the predictions
    % Get the third column- corresponding to 'c' removed, because it is
    % fixed
    preds_vector(mi, :) = predicted_theta_mi;
    preds_vector_arr(mi, :, :) = predicted_theta_mi_mh_arr;
    acts_vector(mi, :) = transpose(fixed_theta_mi);
    
    % Sum the mean squared error predictions over the total number of
    % samples M
    mse_predicted_fixed_theta_sum = mse_predicted_fixed_theta_sum + ...
                         immse(transpose(fixed_theta_mi), predicted_theta_mi.^2);
    
    
                     
    %mse_predicted_sampled_theta = sum((transpose(sampled_theta_mi) - predicted_theta_mi).^2)/5;
    %mse_predicted_sampled_theta_sum = mse_predicted_sampled_theta_sum + mse_predicted_sampled_theta;
    
    % Store the MSE predictions on a per sample basis
    mse_vector(mi,:) = immse(transpose(fixed_theta_mi), predicted_theta_mi.^2);
    %mse_vector(mi,:) = mse_predicted_sampled_theta;
    disp(['MSE for current sample : ', num2str(mse_vector(mi,:))]);
    disp('%====================================================%');
    
end

%==================================================
% Display the final result
%==================================================
avg_mse = mse_predicted_fixed_theta_sum / double(M);
disp(['Estimated MSE using ML based method: ', num2str(avg_mse)]);
disp(['Average time elapsed in secs (per sample) for total process: ', num2str(toc/M)]);

%% Save the results
% res_MC_theta.preds_vector = preds_vector;
% res_MC_theta.acts_vector = acts_vector;
% res_MC_theta.t_elapsed_vec = t_elapsed_vec;
% res_MC_theta.preds_vector_arr = preds_vector_arr;
% res_MC_theta.avgmse = avg_mse;
% res_MC_theta.msevec = mse_vector;
% save('results_ML_fixed_theta_pMCMC.mat', 'res_MC_theta');

res_MC_theta_fixed.preds_vector = preds_vector;
res_MC_theta_fixed.acts_vector = acts_vector;
res_MC_theta_fixed.t_elapsed_vec = t_elapsed_vec;
res_MC_theta_fixed.preds_vector_arr = preds_vector_arr;
res_MC_theta_fixed.avgmse = avg_mse;
res_MC_theta_fixed.msevec = mse_vector;
save('results_ML_sampled_theta_pMCMC_MP100_latest_tuned_indices.mat', 'res_MC_theta_fixed');
