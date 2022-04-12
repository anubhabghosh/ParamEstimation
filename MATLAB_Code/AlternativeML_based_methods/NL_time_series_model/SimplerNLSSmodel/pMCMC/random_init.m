function [theta_i] = random_init(theta_i_orig, p)
    % This function randomly initializes the parameter theta_i_orig with
    % 'p' % of the true value of theta_i_orig. The interval between 50% of
    % the original value is initialized as a uniform distribution
    theta_i = theta_i_orig + (p*theta_i_orig)*(-1 + 2*rand(1));
end