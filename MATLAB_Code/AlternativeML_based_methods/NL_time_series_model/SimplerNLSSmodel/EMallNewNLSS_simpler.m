% This script contains the code necessary for running the nonlinear example 
% in Section 10.2 of the paper
%
% Thomas B. Schön, Adrian Wills and Brett Ninness. System Identification 
% of Nonlinear State-Space Models. Automatica, 47(1):39-49, January 2011. 
%

clear;
clc;
close all;
%rng(100);
% Number of EM iterations
opt.miter = 1000; 
theta = [0.7, 0.8, 1.0, 1.0];
theta_actual = [theta(1), theta(2), theta(3), theta(4)];  
M = 1;
estimates = zeros(M, 4);
for iter=1:M
    % Initialize randomly within 50% of the true value for all parameters
    a0 = theta_actual(1); %+ (0.0*theta_actual(1))*(-1 + 2*rand(1));
    b0 = theta_actual(2); %+ (0.0*theta_actual(2))*(-1 + 2*rand(1));
    Q0 = theta_actual(3) + (0.0*theta_actual(3))*rand(1);
    R0 = theta_actual(4) + (0.0*theta_actual(4))*rand(1);
    theta0 = [a0 b0 sqrt(Q0) sqrt(R0)];    % Initial guess for the parameters
    % Note that we are estimating the square root factor of the covariance in
    % order to enforce the fact that the covariance should be positive
    % semi-definite.
    disp(['Initial parameter guess, a: ' num2str(a0),'  b: ',num2str(b0), '  Q: ',num2str(Q0),'  R: ',num2str(R0)])

    %====================================================
    %===   Simulate the nonlinear state-space model   ===
    %====================================================
    N    = 200;            % Number of data
    m.a  = theta_actual(1);     
    m.b  = theta_actual(2);
    m.Q  = theta_actual(3);    
    m.R  = theta_actual(4);
    m.X1 = 0;       m.P1 = 0;  % Initial state (fully known)
    x = zeros(1,N+1); y = zeros(1,N);
    x(1) = m.X1;
    v = sqrt(m.Q)*randn(1,N);    % Process noise sequence
    e = sqrt(m.R)*randn(1,N);    % Measurement noise sequence
    for t=1:N
      x(t+1) = m.a*x(t)/(1.0+x(t)^2) + cos(1.2*t) + v(t);
      y(t)   = m.b*x(t)^2 + e(t);
    end
    z.y     = y;
    z.xTrue = x(1:N);

    %==================================================
    %===   EM algorithm using a particle smoother   ===
    %==================================================
    % figure;
    % subplot(211);
    % plot(x);
    % title('State sequence');
    % subplot(212);
    % plot(y);
    % title('Observation sequence');
    %==================================================
    %===   EM algorithm using a particle smoother   ===
    %==================================================
    M       = 600;             % Number of particles
    mEst    = m;
    mEst.a  = theta0(1);   
    mEst.b  = theta0(2);   
    mEst.Q  = theta0(3)^2; 
    mEst.R  = theta0(4)^2;
    mEst.X1 = m.X1;            % Initial mean values known
    mEst.P1 = 1;               % Spread the initial particles a bit
    theta   = theta0;
    mEstStore(1) = mEst;       % Store the interates
    %options      = optimset('Display','off','LargeScale','off');
    options      = optimset('LargeScale','off', 'Display','off','TolFun',1e-5,'TolX',1e-5); 
    for k = 1:opt.miter
      % E step
      gPF = pf_new_simpler(mEst,M,z);      % Particle filter
      gPS = ps_new_simpler(mEst,M,gPF);    % Particle smoother

      % M step
      [theta,Qval] = fminunc(@(theta) nl1NoGrad_new_simpler(theta,gPS.xPart, z.y),theta,options);
      mEst.a = theta(1);   mEst.b = theta(2);
      mEst.Q = theta(3).^2;   mEst.R = theta(4).^2;
      mEstStore(k) = mEst;    % Store the iterates
      if mod(k, 1) == 0
        disp(['Iteration nr: ' num2str(k) ' Estimates, a: ' num2str(mEst.a),'  b: ',num2str(mEst.b),'  Q: ',num2str(mEst.Q),'  R: ',num2str(mEst.R)]);
      end
    end
    disp('----------------------------------------------------------------------------------------------------------');
    disp(['Iteration nr: ' num2str(k) ' Estimates, a: ' num2str(mEst.a),'  b: ',num2str(mEst.b),'  Q: ',num2str(mEst.Q),'  R: ',num2str(mEst.R)]);
    disp('----------------------------------------------------------------------------------------------------------');
    estimates(iter, :) = [mEst.a mEst.b mEst.Q mEst.R];
end
