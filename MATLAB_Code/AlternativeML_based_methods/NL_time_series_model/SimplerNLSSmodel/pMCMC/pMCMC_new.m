% This function should be able to return a chain of values of theta by 
% using a Metropolis-hastings algorithm for sampling from the posterior
% distribution. The following parameters are rqeuired to be provided:
% z - dataset (as a structure, data is present as z.y)
% m - structure containing the parameters that are to be sent to pf()
% M - number of particles in the particle filter
% theta_0 - vector of parameters that represent the 'actual' theta
% T - total number of iterations to run pMCMC (this includes burn in period
% + actual number of samples)

function [theta_samples] = pMCMC_new(z, M, theta_0, alpha, T)

% Initialise the initial value of phi within 50 % of the true parameter
% theta_0

% % Fixed theta
% a0_init = random_init(theta_0(1),0.1);
% b0_init = random_init(theta_0(2),0.1);
% c0_init = 8.0; % Usually not globally identifiable
% d0_init = random_init(theta_0(4), 0.0);
% Q0_init = random_init_var(theta_0(5),0.1); % Variance Q
% R0_init = random_init_var(theta_0(6),0.1); % Variance R

% Sampled theta
a0_init = random_init(theta_0(1),0.0);
b0_init = random_init(theta_0(2),0.0);
Q0_init = random_init_var(theta_0(3),0.0); % Variance Q
R0_init = random_init_var(theta_0(4),0.0); % Variance R

% true parameter with 'true' values of theta
% NOTE: we initialize theta_init for Q, R to be the square root of the
% true parameters
%theta0_init_true = [theta_0(1) theta_0(2) theta_0(3) theta_0(4) theta_0(5) theta_0(6)];
theta0_init_sampled = [a0_init b0_init Q0_init R0_init]; 


%======================================================================
% Initialize MH related parameters
% assumption: proposal distribution is Gaussian with diagonal
% covariances, assuming zero mean
% NOTE: y (or equivalently z) remains kind of fixed, theta is to be 
% generated with MH
%======================================================================

theta_samples = zeros(T, 4);
phi_samples = zeros(T, 4);

%=========================================================================%
% Initialization of theta, phi parameters
%=========================================================================%
%theta_samples(1, :) = theta0_init_true; % Option 1: Initialize with true parameter value first!
theta_samples(1, :) = theta0_init_sampled; % Option 2: Initialize with sampled parameter value first!
phi_samples(1, :) = g_vector(theta_samples(1, :));
%theta_samples(1, :) = theta0_init_sampled;
disp('Initial theta:');
disp([theta_samples(1, 1), theta_samples(1, 2), theta_samples(1, 3).^2, theta_samples(1, 4).^2]);

%cov_matrix = diag(alpha * ones(1, 6));

% Choice of standard deviations are hard-coded (for pilot simulation)
std_matrix = diag(sqrt([alpha/150 alpha alpha/150 alpha/200]));

% Using parameters obtained from a pilot simulation run as per the used
% formula: 2.565^2/3 * var(theta_pilotRun) 
% std_matrix = diag(sqrt([0.0035    0.05    0.0033    0.0018]));

%flag_within_prior = false;

%==============================================================================%
% Start the Markov chain with samples from i=2, 3, 4, ..., T
%==============================================================================%
A_count = 0;
R_count = 0;
m_theta_proposed_init.a = theta_samples(1, 1); m_theta_proposed_init.b = theta_samples(1, 2); 
m_theta_proposed_init.Q = theta_samples(1, 3).^2; 
m_theta_proposed_init.R = theta_samples(1, 4).^2;
m_theta_proposed_init.X1 = 0;       m_theta_proposed_init.P1 = 0;  % Initial state (fully known)

[~, approxLogLik_theta_prev] = pf_new(m_theta_proposed_init, M, z);

for i=2:T
    %==============================================================================%
    % Sample from the proposal distribution and check whether the
    % sample is within the support of the uniform prior, if not we
    % should re-sample
    %==============================================================================%
    A = compute_cholesky(std_matrix.^2);
    
    % Using parameters obtained from a pilot simulation run as per the used
    % formula: 2.565^2/3 * empirical_cov(theta_pilotRun) would yield the
    % Cholesky factor of the std_matrix for the proposal distribution
    %std_matrix = [0.0002   -0.0036         0   -0.0000   -0.0001   -0.0003;
    %  -0.0036    0.5838         0   -0.0000   -0.0007    0.0276;
    %   0         0         0         0         0         0;
    %  -0.0000   -0.0000         0    0.0000    0.0000    0.0000;
    %  -0.0001   -0.0007         0    0.0000    0.0060    0.0029;
    %  -0.0003    0.0276         0    0.0000    0.0029    0.0104];
    
    phi_proposed = phi_samples(i-1, :) + randn(1, 4) * A;
    theta_proposed = h_vector(phi_proposed);
    
    m_theta_proposed.a = theta_proposed(1); m_theta_proposed.b = theta_proposed(2); 
    m_theta_proposed.Q = theta_proposed(3).^2; m_theta_proposed.R = theta_proposed(4).^2;
    %m_theta_proposed.Q = theta_proposed(3); m_theta_proposed.R = theta_proposed(4);
    m_theta_proposed.X1 = 0;       m_theta_proposed.P1 = 0;  % Initial state (fully known)
    
    [~, approxLogLik_theta_proposed] = pf_new(m_theta_proposed, M, z);
    
    % Compute the acceptance rate (after reparameterization)
    log_acceptance_rate = approxLogLik_theta_proposed - approxLogLik_theta_prev + ...
           sum(phi_proposed - phi_samples(i-1, :) + ...
           2.*log(exp(phi_samples(i-1, :)) + 1) - 2.*log(exp(phi_proposed) + 1));
       
    % Check if the acceptance rate is more or less
    % If greater: Accept new sample, else copy old sample
    if log_acceptance_rate > log(rand(1,1))
        %disp("Accepted, and falls within prior!");
        theta_samples(i, :) = [m_theta_proposed.a m_theta_proposed.b sqrt(m_theta_proposed.Q) sqrt(m_theta_proposed.R)];
        phi_samples(i, :) = g_vector(theta_samples(i, :));
        A_count = A_count + 1;
        approxLogLik_theta_prev = approxLogLik_theta_proposed;

    else
        %disp("Rejected, but falls within of prior!");
        theta_samples(i, :) = theta_samples(i-1, :);
        phi_samples(i, :) = phi_samples(i-1, :);
        R_count = R_count + 1;
    end
    % Displaying the values every iteration
    if mod(i-1, 500) == 0
        disp(['Iteration nr: ' num2str(i-1) ' Estimates, a: ' num2str(theta_samples(i,1)),'  b: ', ...
            num2str(theta_samples(i,2)),'  Q: ',num2str(theta_samples(i,3).^2),'  R: ',num2str(theta_samples(i,4).^2)])
        disp(['Accept count (normalized):', num2str(A_count ./ (A_count+R_count)), ', Reject count (normalized): ', num2str(R_count ./ (R_count+A_count))]);
        disp(['logA :', num2str(log_acceptance_rate)]);
        %pause(0.0);
    end
end 
formatspec = "Final Accept rate (after %d iterations) : %.4f, Reject rate: %.4f \n";
fprintf(formatspec, T, A_count ./ (A_count+R_count), R_count ./ (A_count+R_count));
end

function [phi_vector] = g_vector(theta_vector)
    
    phi_vector = zeros(1, length(theta_vector));
    phi_vector(1) = g(theta_vector(1), 0.0, 1.0);
    phi_vector(2) = g(theta_vector(2), 0.1, 2.0);
    phi_vector(3) = g(theta_vector(3), eps, 1.0);
    phi_vector(4) = g(theta_vector(4), eps, 1.0);
    
end

function [theta_vector] = h_vector(phi_vector)
    theta_vector = zeros(1, length(phi_vector));
    
    theta_vector(1) = h(phi_vector(1), 0.0, 1.0);
    theta_vector(2) = h(phi_vector(2), 0.1, 2.0);
    theta_vector(3) = h(phi_vector(3), eps, 1.0);
    theta_vector(4) = h(phi_vector(4), eps, 1.0);
    
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
