clear;
clc;
close all;
%====================================================
%==================  True Model  ====================
%====================================================
a0 = 0.7;
b0 = 1.0;
Q0 = 0.1;
R0 = 0.1;
theta0 = [a0 b0 sqrt(Q0) sqrt(R0)];  % true parameter

%====================================================
%===   Simulate the nonlinear state-space model   ===
%====================================================
N    = 200;            % Number of data
m.a  = theta0(1);     
m.b  = theta0(2);
m.Q  = theta0(3).^2;    
m.R  = theta0(4).^2;
m.X1 = 0;       m.P1 = 0;  % Initial state (fully known)
x = zeros(1,N+1); y = zeros(1,N);
x(1) = m.X1;
v = sqrt(m.Q)*randn(1,N);    % Process noise sequence
e = sqrt(m.R)*randn(1,N);    % Measurement noise sequence
num_particles = 10000; % number of particles in the PF

u = cos(1.2.*linspace(1, N, N));  % Cosine input
%u = 2*randn(1, N); % Random input signal
for t=1:N
  %x(t+1) = m.a*x(t)/(1.0+x(t)^2) + 8*cos(1.2*t) + v(t);
  %x(t+1) = m.a*x(t)/(1.0+ x(t)^2) + u(t) + v(t);
  x(t+1) = m.a*x(t)/(1.0+ (0.2.*x(t)).^2) + u(t) + v(t);
  y(t)   = m.b*x(t).^2 + e(t);
end
z.y     = y;
z.xTrue = x(1:N);
z.u = u;
% %==================================================
% %===   pMCMC via auxilary PF and MH algorithm   ===
% %==================================================
%=====================================================================
alpha = 0.45; % Tunable parameter for MH algorithm
%alpha_arr = [0.01];% 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6];
% T = T_burnin + T_required % Only start considering after burn-in number
% of samples
% Thinning: Skip l number of samples, post the burn-in samples, so this
% requires more simulations, to improve the variance
T_burnin = 10^3;
T_required = 3500;
T = T_burnin + T_required;

% Number of samples to skip post burn in samples to store the states of 
% the Markov chain
thinning_interval = 10; 
    
%=====================================================================
formatspec = "%s theta vector: a: %.4f, b: %.4f, Q: %.4f, R: %.4f \n"; 
disp('------------------------------------------------------------------------------------');
disp(['Alpha: ', num2str(alpha)]);
disp(['Number of MH iterations: ', num2str(T)]);

% returns a markov chain, now using the new reparameterized computation
theta_samples = pMCMC_new(z, num_particles, theta0, alpha, T);  

% estimate of conditional mean
thetaHat = mean(theta_samples(T_burnin:end, :));  % an estimate of E[theta|Y ] = conditional mean, only after samples for the burn-in period
fprintf(formatspec, "True", theta0(1), theta0(2), theta0(3).^2, theta0(4).^2); 
disp('------------------------------------------------------------------------------------');
fprintf(formatspec, "Estimated", thetaHat(1), thetaHat(2), thetaHat(3).^2, thetaHat(4).^2);
mse = immse([theta0(1), theta0(2), theta0(3).^2, theta0(4).^2], ...
    [thetaHat(1), thetaHat(2), thetaHat(3).^2, thetaHat(4).^2]);
disp(['MSE: ', num2str(mse)]);
disp('------------------------------------------------------------------------------------');
%end

