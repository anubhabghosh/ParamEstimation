% This script contains the function file for calling the Maximum
% likelihood estimation function based on conditional particle filter
% (using expectation maximization). Mainly functions as a kind of wrapper
% script. Some important distinctions w.r.t. 'evaluate_using_NLSS.m' are the
% following: 

% 1. The trajectories 'y' used for estimation have been generated using a
% Python based code that simulates the non-linear state space model BUT
% with the choice of theta being RANDOM in each simulation of the
% trajectory.

% 2. The 'true' parameter 'theta_0' is NOT going to be initialized as the
% vector [0.7, 1.0, 0.1, 0.1], or within 20 - 50 % of the true value as
% in the previous code or original function 'EMallNLSS.m'. 

% 3. The last two parameters Q = 0.1 amd R = 0.1 model the variances
% directly. So a particular theta vector from the Python code looks like
% this [a, b, Q, R]. However, the EMallNLSS method estimates the 
% theta vector as the [a, b, sqrt(Q), sqrt(R)] for enforcing 
% positive semi-definite nature of the covariances

function predicted_theta_vector = evaluate_using_NLSS_fixed_theta_simpler_model(y, theta_actual, N)
    
    %======================================================================
    % Initialize randomly within 50% of the true value for all parameters
    %======================================================================
    % Initialize randomly within 50% of the true value for all parameters
    % Initialize randomly within 50% of the true value for all parameters
    a0 = theta_actual(1) + (0.2*theta_actual(1))*(-1 + 2*rand(1));
    b0 = theta_actual(2) + (0.2*theta_actual(2))*(-1 + 2*rand(1));
    Q0 = theta_actual(3) + (0.2*theta_actual(3))*rand(1);
    R0 = theta_actual(4) + (0.2*theta_actual(4))*rand(1);
    theta0 = [a0 b0 sqrt(Q0) sqrt(R0)];    % Initial guess for the parameters
    
    disp(['Initial parameter guess, a: ' num2str(a0),'  b: ',num2str(b0), ...
        '  Q: ',num2str(Q0),'  R: ',num2str(R0)])
    
    %======================================================================
    % EM algorithm using a particle smoother
    %======================================================================
    m.a  = theta_actual(1);     
    m.b  = theta_actual(2);
    m.Q  = theta_actual(3);    
    m.R  = theta_actual(4);
    m.X1 = 0;       m.P1 = 0;  % Initial state (fully known)
    %x = zeros(1,N+1); y = zeros(1,N);
    %x(1) = m.X1;
    %v = sqrt(m.Q)*randn(1,N);    % Process noise sequence
    %e = sqrt(m.R)*randn(1,N);    % Measurement noise sequence
    %for t=1:N
    %  x(t+1) = m.a*x(t)/(1.0+x(t)^2) + cos(1.2*t) + v(t);
    %  y(t)   = m.b*x(t)^2 + e(t);
    %end
    u = cos(1.2.*(linspace(1, N, N)));  % Cosine input
    z.u = u;
    z.y = y;
    %z.xTrue = x(1:N);
    M       = 200;             % Number of particles
    mEst    = m;
    mEst.a  = theta0(1);   
    mEst.b  = theta0(2);   
    mEst.Q  = theta0(3)^2; 
    mEst.R  = theta0(4)^2;
    mEst.X1 = m.X1;            % Initial mean values known
    mEst.P1 = 1;               % Spread the initial particles a bit
    theta   = theta0;
    mEstStore(1) = mEst;      
    % Number of EM iterations
    opt.miter = 1000; 
    options      = optimset('LargeScale','off', 'Display','off','TolFun',1e-5,'TolX',1e-5); 
    for k = 1:opt.miter
      % E step
      gPF = pf_new_simpler(mEst,M,z);      % Particle filter
      gPS = ps_new_simpler(mEst,M,gPF,z);    % Particle smoother

      % M step
      [theta,Qval] = fminunc(@(theta) nl1NoGrad_new_simpler(theta,gPS.xPart, z),theta,options);
      mEst.a = theta(1);   
      mEst.b = theta(2);
      mEst.Q = theta(3).^2;   
      mEst.R = theta(4).^2;
      
      mEstStore(k) = mEst;    % Store the iterates
      if mod(k, 200) == 0
        disp(['Iteration nr: ' num2str(k) ' Estimates, a: ' num2str(mEst.a),'  b: ',num2str(mEst.b),'  Q: ',num2str(mEst.Q),'  R: ',num2str(mEst.R)]);
      end
    end
    disp('----------------------------------------------------------------------------------------------------------');
    disp(['Iteration nr: ' num2str(k) ' Estimates, a: ' num2str(mEst.a),'  b: ',num2str(mEst.b),'  Q: ',num2str(mEst.Q),'  R: ',num2str(mEst.R)]);
    disp('----------------------------------------------------------------------------------------------------------');
    predicted_theta_vector = [mEst.a mEst.b mEst.Q mEst.R];
    
    
    
    
end