clear;
clc;
close all;

%=====================================================%
% Declare a parameter \alpha to control the support of 
% the uniform proposal distribution
%=====================================================%
alpha = 1.0;
%=====================================================%
% Define lambda style funs for sampling and llh
% of the uniform proposal pdf
%=====================================================%
%proposal_pdf_log_likelihood = @(y, x, alpha) log(unifpdf(y, x-alpha, x+alpha)+eps);
%proposal_pdf_sample = @(x, alpha) x-alpha + (2*alpha)*rand(0,1);

%=====================================================%
% Run metroplis algorithm
T = 10^4;
x = repmat(pi, T, 1);
for i=2:T
    %disp(['Iteration:',num2str(i)]);
    x(i) = metropolis_sampling(x(i-1), alpha);
    pause(0.0);
end

x_in = linspace(-pi, pi, T)';
px_in = zeros(length(x_in), 1);
for i=1:length(px_in)
    px_in(i) = compute_target_llh(x_in(i));
end

target_pdf_fn = @(x) (sin(x)).^2 .* (sin(2.*x)).^2 .* exp(-0.5.*(x.^2)) .* (1.0/sqrt(2.*pi));
normalization_constant = integral(target_pdf_fn, -pi, pi);

figure;
plot(x_in, px_in ./ normalization_constant, 'b', 'linewidth', 2);
hold on;
histogram(x, 'Normalization', 'pdf');
legend('actual', 'estimated');
hold off;



