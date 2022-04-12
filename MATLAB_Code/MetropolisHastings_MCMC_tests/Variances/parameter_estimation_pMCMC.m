clear;
clc;
close all;
%====================================================
%==================  True Model  ====================
%====================================================
a0 = 0.5;
b0 = 25;
c0 = 8;     % not globally identifiable
d0 = 0.05;
Q0 = 1.0;
R0 = 0.1;
theta0 = [sqrt(Q0) sqrt(R0)];  % true parameter

%====================================================
%===============  Simulate Data Model  ==============
%====================================================
N    = 200;            % Number of data
M = 1000; % number of particles in the PF
%m.a  = theta0(1);     m.b  = theta0(2);
%m.c  = theta0(3);       m.d  = theta0(4);
m.Q  = theta0(1).^2;    m.R  = theta0(2).^2;
m.X1 = 0;       m.P1 = 0;  % Initial state (fully known)
x = zeros(1,N+1); y = zeros(1,N);
x(1) = m.X1;
v = sqrt(m.Q)*randn(1,N);    % Process noise sequence
e = sqrt(m.R)*randn(1,N);    % Measurement noise sequence
for t=1:N
   x(t+1) = 0.5*x(t) + 25.0*x(t)/(1+x(t)^2) + 8.0*cos(1.2*t) + v(t);
   y(t)   = 0.05*x(t)^2 + e(t);
%  x(t+1) = 0.5*x(t) + 8.0*cos(1.2*t) + v(t);
%  y(t)   = 0.05*x(t) + e(t);
end
z.y = y;  % data set
% %==================================================
% %===   pMCMC via auxilary PF and MH algorithm   ===
% %==================================================
%=====================================================================
alpha = 1.0; % Tunable parameter for MH algorithm
%alpha_arr = [0.01];% 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6];
% T = T_burnin + T_required % Only start considering after burn-in number
% of samples
% Thinning: Skip l number of samples, post the burn-in samples, so this
% requires more simulations, to improve the variance
T_burnin = 1e3;
T_required = 3.5e3;
T = T_burnin + T_required;

% Number of samples to skip post burn in samples to store the states of 
% the Markov chain
thinning_interval = 10; 
    
%=====================================================================
formatspec = "%s theta vector: Q: %.4f, R: %.4f \n"; 
disp('------------------------------------------------------------------------------------');
disp(['Alpha: ', num2str(alpha)]);
disp(['Number of MH iterations: ', num2str(T)]);

% returns a markov chain, now using the new reparameterized computation
theta_samples = pMCMC_new(z, M, theta0, alpha, T);  

%% estimate of conditional mean
formatspec = "%s theta vector: Q: %.4f, R: %.4f \n"; 
thetaHat = mean(theta_samples(T_burnin:end, :));  % an estimate of E[theta|Y ] = conditional mean, only after samples for the burn-in period
fprintf(formatspec, "True", theta0(1).^2, theta0(2).^2); 
disp('------------------------------------------------------------------------------------');
fprintf(formatspec, "Estimated", thetaHat(1).^2, thetaHat(2).^2);
mse = immse([theta0(1).^2, theta0(2).^2], ...
    [thetaHat(1).^2, thetaHat(2).^2]);
disp(['MSE: ', num2str(mse)]);
disp('------------------------------------------------------------------------------------');

%% Plot the Markov chain results
figure;
subplot(121);
histogram(theta_samples(:, 1).^2);title('Q');
subplot(122);
histogram(theta_samples(:, 2).^2);title('R');
%end

figure;
subplot(211);
plot(theta_samples(:, 1).^2);title('Q');
subplot(212);
plot(theta_samples(:, 2).^2);title('R');

