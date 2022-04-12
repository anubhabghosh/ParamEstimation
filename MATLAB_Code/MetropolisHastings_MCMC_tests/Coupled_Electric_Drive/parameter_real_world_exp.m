clear;
clc;
close all;
%% ==================  True Model  ====================
%====================================================
% load data
load('DATAPRBS.mat')
Ts = 20e-3;
figure(1)
subplot(2,1,2)
plot((1:500).*Ts, u1)
grid on
subplot(2,1,1)
plot((1:500).*Ts, z1)
grid on
hold all

%====================================================
%===============  Simulate Data Model  ==============
%====================================================
s = tf('s');

% possible model  (table 2 in section 3.4, second row)
a =  5163;
b = -19.93;
c = -509.8;
d = -2835;
G = a/(s^3 -b*s^2 - c*s -d);


k = -a/d;             % a/d;
alpha = -6.73507; % obtained by solving qubic equation(roots([1 -b -c -d])) 
                  % alpha^3 + b alpha^2 + c alpha +d = 0  
                  % take real solution
alpha = -alpha;
w0 = sqrt(-d/alpha);%w0 = sqrt(d/alpha);     %
xi = (-b - alpha)/(2*w0);               %(b-alpha)/(2*w0);
Gtilde = (k*alpha*w0^2)/((s+alpha)*(s^2+2*xi*w0*s + w0^2));  % same as G
disp(["Is constructed transfer function stable?", num2str(isstable(Gtilde))]);

%% Figure: bode plot
figure(2)
bode(G)
y_model = abs(lsim(G,u1,(0:500-1).*Ts));
figure(1)
subplot(2,1,1)
plot((1:500).*Ts, y_model)

%% Get the limits to the MH prior
percent_ = 0.2; % Sample within 20% of the prior
prbs_struct = load('prbs_dataset_opt.mat');
[k_llim, alpha_llim, w0_llim, xi_llim] = get_unif_lower_limit(prbs_struct, percent_);
[k_ulim, alpha_ulim, w0_ulim, xi_ulim] = get_unif_lower_limit(prbs_struct, percent_);
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

