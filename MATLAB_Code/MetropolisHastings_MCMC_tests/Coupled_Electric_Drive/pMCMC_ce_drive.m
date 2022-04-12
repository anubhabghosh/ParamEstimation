% This function should be able to return a chain of values of theta by 
% using a Metropolis-hastings algorithm for sampling from the posterior
% distribution. The following parameters are rqeuired to be provided:
% z - dataset (as a structure, data is present as z.y)
% m - structure containing the parameters that are to be sent to pf()
% M - number of particles in the particle filter
% theta_0 - vector of parameters that represent the 'actual' theta
% T - total number of iterations to run pMCMC (this includes burn in period
% + actual number of samples)

function [theta_samples] = pMCMC_ce_drive(z, theta_0, prbs_dataset_opt, alpha, T, percent_, Ts)

% Initialise the initial value of phi within 50 % of the true parameter
% Sampled theta
%a0_init = random_init(theta_0(1),0.0);
%b0_init = random_init(theta_0(2),0.0);
%c0_init = 8.0; % Usually not globally identifiable
%d0_init = random_init(theta_0(4), 0.0);
%Q0_init = random_init_var(theta_0(1),0.0); % Variance Q
%R0_init = random_init_var(theta_0(2),0.0); % Variance R

% true parameter with 'true' values of theta
% NOTE: we initialize theta_init for Q, R to be the square root of the
% true parameters
theta0_init_true = [theta_0(1) theta_0(2) theta_0(3) theta_0(4) theta_0(5)];
%theta0_init_sampled = [theta_0(1) theta_0(2) theta_0(3) theta_0(4) theta_0(5)]; 
P = size(theta0_init_true, 2);

%=========================================================================%
% Initialize MH related parameters
% assumption: proposal distribution is Gaussian with diagonal
% covariances, assuming zero mean
% NOTE: y (or equivalently z) remains kind of fixed, theta is to be 
% generated with MH
%=========================================================================%
theta_samples = zeros(T, P);
phi_samples = zeros(T, P);

%=========================================================================%
% Initialization of theta, phi parameters
%=========================================================================%
%====== Option 1: Initialize with true parameters ======%
theta_samples(1, :) = theta0_init_true; 

%====== Option 2: Sampled from a prior around the true parameters ======%
%theta_samples(1, :) = theta0_init_sampled; 

% For the chosen initial value we get a initial value of the transformed
% domain phi_samples
phi_samples(1, :) = g_vector(theta_samples(1, :), prbs_dataset_opt, percent_);

% Display the initial choice
disp('Initial theta:');
disp([theta_samples(1, 1), theta_samples(1, 2), theta_samples(1, 3), ...
    theta_samples(1, 4), theta_samples(1, 5).^2]);


%=========================================================================%
% Choice of the standard deviation values of the parameters
% NOTE: Using parameters obtained from a pilot simulation run as per the 
% used formula: 2.565^2/3 * empirical_var(theta_pilotRun)
%=========================================================================%

% Choice of standard deviations are heuristically chosen for now
%std_matrix = diag(sqrt([alpha/2 alpha alpha/2 alpha alpha/2e4])); % heuristic

std_matrix = diag(sqrt([alpha*5 alpha*10 alpha/5 alpha*10 alpha/1e4])); % heuristic

%std_matrix = diag(sqrt([0.1 0.5 1.0 0.1 1e-6])); % pilot
%flag_within_prior = false;

%==============================================================================%
% Start the Markov chain with samples from i=2, 3, 4, ..., T
%==============================================================================%
A_count = 0;
R_count = 0;

% We get the discretized transfer function (that basically should reflect
% the matrices A_{d}, B_{d} in the state transition equation) 

[~, x_model_discrete] = get_discr_model(z.u, theta_samples(1, 1), theta_samples(1, 2), theta_samples(1, 3), ...
    theta_samples(1, 4), Ts);

z.x = x_model_discrete;
Var_e = theta_samples(1, end).^2;

approxLogLik_theta_prev = get_mvn_llh(z.y, z.x, Var_e);

for i=2:T
    
    %=====================================================================%
    % 1. Sample from the reparametrized distribution Phi
    % 2. Convert the sampled phi vector to the theta vector
    % 3. Using the resultant theta vector, get the k, alpha, w0, xi and
    % sigma_e. 
    % 4. Build a ss model by first formulating the CT transfer function 
    % and then convert the CT transfer function to DT by c2d
    % 5. Then simulate the resulting DT model by lsim, store x (state
    % sequence)
    % 6. Then compute the llh using y, x and sigma_e obtained from the Markov
    % chain
    %==============================================================================%
    % Sample from the proposal distribution and check whether the
    % sample is within the support of the uniform prior, if not we
    % should re-sample
    %==============================================================================%
    
    %==============================================================================%
    % 1. Sample from the reparametrized distribution Phi
    A = compute_cholesky(std_matrix.^2);
    phi_proposed = phi_samples(i-1, :) + randn(1, P) * A;
    %==============================================================================%
    
    %==============================================================================%
    % 2. Convert the sampled phi vector to the theta vector
    theta_proposed = h_vector(phi_proposed, prbs_dataset_opt, percent_);
    %==============================================================================%
    
    %==============================================================================%
    
    % 3. Using the resultant theta vector, get the k, alpha, w0, xi and
    % sigma_e. 
    % 4. Build a ss model by first formulating the CT transfer function 
    % and then convert the CT transfer function to DT by c2d
    % 5. Then simulate the resulting DT model by lsim, store x (state
    % sequence). The variance var_e is just in the likelihood calc.
    % function
    [~, x_model_discrete_proposed] = get_discr_model(z.u, theta_proposed(1), theta_proposed(2), ...
        theta_proposed(3), theta_proposed(4), Ts);

    z.x = x_model_discrete_proposed;
    Var_e_proposed = theta_proposed(end).^2;
    %==============================================================================%
    
    %==============================================================================%
    % 6. Then compute the llh using y, x and sigma_e obtained from the Markov
    % chain
    approxLogLik_theta_proposed = get_mvn_llh(z.y, z.x, Var_e_proposed);
    %==============================================================================%
    
    %==============================================================================%
    % 7. Compute the acceptance rate (after reparameterization)
    %==============================================================================%
    log_acceptance_rate = approxLogLik_theta_proposed - approxLogLik_theta_prev + sum(phi_proposed - phi_samples(i-1, :) + 2.*log(exp(phi_samples(i-1, :)) + 1) - 2.*log(exp(phi_proposed) + 1));
    dformatspec = "Approx loglik prop. : %.4f, Approx loglike_prev. : %.4f, Sum term: %.4f \n";
    assert(isnan(log_acceptance_rate) == false, 'NaN encountered approx loglik prop. : %.4f, approx loglike_prev. : %.4f, sum term: %.4f', ...
        approxLogLik_theta_proposed, approxLogLik_theta_prev, sum(phi_proposed - phi_samples(i-1, :) + ...
           2.*log(exp(phi_samples(i-1, :)) + 1) - 2.*log(exp(phi_proposed) + 1)));
    %==============================================================================%
    % 8. Accept or Reject the proposed value 

    % Check if the acceptance rate is more or less
    % If greater: Accept new sample, else copy old sample
    if log_acceptance_rate > log(rand(1,1))
        % Accepting the proposed value and including it into the chain
        %disp("Accepted, and falls within prior!");
        theta_samples(i, :) = [theta_proposed(1) theta_proposed(2) theta_proposed(3) ...
            theta_proposed(4) theta_proposed(5)];
        phi_samples(i, :) = g_vector(theta_samples(i, :), prbs_dataset_opt, percent_);
        A_count = A_count + 1;
        approxLogLik_theta_prev = approxLogLik_theta_proposed;

    else
        % Rejecting the value, so copying the previous value in the chain
        %disp("Rejected, but falls within of prior!");
        theta_samples(i, :) = theta_samples(i-1, :);
        phi_samples(i, :) = phi_samples(i-1, :);
        R_count = R_count + 1;
    end
    %==============================================================================%
    
    % Displaying the values every iteration
%     if mod(i-1, 1000) == 0
    if mod(i-1, 10) == 0
        disp(['Iteration nr: ' num2str(i-1) ...
            ' Estimates, k: ',num2str(theta_samples(i,1)),'  alpha: ',num2str(theta_samples(i,2)), ...
            ' omega0: ',num2str(theta_samples(i,3)),'  xi: ',num2str(theta_samples(i,4)), ...
            ' e_k_var: ',num2str(theta_samples(i,5).^2)]);
        disp(['Accept count (normalized):', num2str(A_count ./ (A_count+R_count)), ', Reject count (normalized): ', num2str(R_count ./ (R_count+A_count))]);
        disp(['logA :', num2str(log_acceptance_rate)]);
        %fprintf(dformatspec, approxLogLik_theta_proposed, approxLogLik_theta_prev, sum(phi_proposed - phi_samples(i-1, :) + ...
        %   2.*log(exp(phi_samples(i-1, :)) + 1) - 2.*log(exp(phi_proposed) + 1)));
        %pause(0.5);
    end
end 
formatspec = "Final Accept rate (after %d iterations) : %.4f, Reject rate: %.4f \n";
fprintf(formatspec, T, A_count ./ (A_count+R_count), R_count ./ (A_count+R_count));
end

function [phi_vector] = g_vector(theta_vector, prbs_struct, percent_)
    phi_vector = zeros(1, length(theta_vector));
    [k_ulim, alpha_ulim, w0_ulim, xi_ulim] = get_unif_upper_limit(prbs_struct, percent_);
    [k_llim, alpha_llim, w0_llim, xi_llim] = get_unif_lower_limit(prbs_struct, percent_);
    phi_vector(1) = g(theta_vector(1), k_llim, k_ulim); % k 
    phi_vector(2) = g(theta_vector(2), alpha_llim, alpha_ulim); % alpha 
    phi_vector(3) = g(theta_vector(3), w0_llim, w0_ulim); % w0 
    phi_vector(4) = g(theta_vector(4), xi_llim, xi_ulim); % xi 
    phi_vector(5) = g(theta_vector(5), sqrt(eps), sqrt(0.01)); % [eps, 0.01] Variance E, theta_vector(5) tries model the std. dev
    %assert(isreal(phi_vector) == true, ...
    %    'proposed (parameterized.) phi is not real anymore! phi_1: %.4f, phi_2: %.4f, phi_3: %.4f, phi_4: %.4f, phi_5: %.4f', ...
    %    phi_vector(1), phi_vector(2), phi_vector(3), phi_vector(4), phi_vector(5));
    
end

function [theta_vector] = h_vector(phi_vector, prbs_struct, percent_)
    theta_vector = zeros(1, length(phi_vector));
    [k_ulim, alpha_ulim, w0_ulim, xi_ulim] = get_unif_upper_limit(prbs_struct, percent_);
    [k_llim, alpha_llim, w0_llim, xi_llim] = get_unif_lower_limit(prbs_struct, percent_);
    theta_vector(1) = h(phi_vector(1), k_llim, k_ulim); % k 
    theta_vector(2) = h(phi_vector(2), alpha_llim, alpha_ulim); % alpha 
    theta_vector(3) = h(phi_vector(3), w0_llim, w0_ulim); % w0 
    theta_vector(4) = h(phi_vector(4), xi_llim, xi_ulim); % xi 
    theta_vector(5) = h(phi_vector(5), sqrt(eps), sqrt(0.01)); % [eps, 0.01] Variance E, theta_vector(5) tries model the std. dev
    %assert(isreal(theta_vector) == true, 'proposed theta is not real anymore!')
    
end

function [R] = compute_cholesky(Cov_matrix)
    % This means that Cov_matrix = R * R', since by default MATLAB computes
    % this as Cov_matrix = R' * R, where R = chol(Cov_matrix)
    R = chol(Cov_matrix)';
end

function [phi] = g(theta, a, b)
    phi = log((theta - a + eps) ./ (b - theta + eps));
end

function [theta] = h(phi, a, b)
    theta = (b.*exp(phi) + a) ./ (exp(phi) + 1);
end
