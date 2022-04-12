rng(100)
close all

N = 100;
M = 10000; % number of particles in the PF

x0 = 0;  % initial state

% model parameter
theta = 0.5;
sigmaw = 0.5;
sigmav = 0.1;

% making some room
y = zeros(N,1);
x = zeros(N+1,1);


% generating a data set
w = sigmaw*randn(N,1);
v = sigmav*randn(N,1);

% computing the initial conditions
x(1) =  x0; % known initial condition
y(1) =  x(1) + v(1);


% simulating the model
% get output trajectory y and the true state x
for t = 2:N
    x(t) = theta*x(t-1) + w(t-1);
    y(t)   = x(t) + v(t);
end
% ===============================================

% Goal given the model with known parameters, and a dataset (= y)
% compute the log likelihood function >>  log p(y; parameters)

% analytitcal log likelihood

F = @(theta) toeplitz([0 1 (theta*ones(1,N-2)).^[1:N-2]], zeros(1,N));
Sigma = F(theta)*F(theta)'*sigmaw^2 + sigmav^2*eye(N);  % covariance of Y

logLik_builtin =  log(mvnpdf(y,zeros(N,1),Sigma))
logLik = -0.5*y'*inv(Sigma)*y - (N/2)*log(2*pi) - 0.5*log(det(Sigma))


% posterior mean of theta
% E[theta|Y ] = conditional mean


% PF
[g, approxLogLik] = pf(m,M,iddata(y));
% g.Xf filtered state

% KF
xHatFiltered_KF = kalmanFilter(y, [theta; sigmaw;sigmav], x0, 0);
error_pf_kf = norm(xHatFiltered_KF-g.Xf');

plot(x(1:end-1))
hold all
plot(g.Xf)
plot(xHatFiltered_KF)
legend('true','pf', 'kf')

disp(['Analytitcal log likelihood = ', num2str(logLik)])
disp(['Estimated log likelihood = ', num2str(approxLogLik)])
disp(['Error in pf estimation of log likelihood = ', num2str(logLik-approxLogLik)])
disp(['Error in pf estimation of state = ', num2str(error_pf_kf)])
disp(['State estimation error kf = ', num2str(norm(x(1:end-1)-xHatFiltered_KF))])


