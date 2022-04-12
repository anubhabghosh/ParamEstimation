function [theta_i] = random_init_var(theta_i_orig, p)
    % This function randomly initializes the parameter theta_i_orig with
    % 'p' % of the true value of theta_i_orig. The interval between 50% of
    % the original value is initialized as a uniform distribution between 0
    % and 1
    theta_i = theta_i_orig + (p*theta_i_orig)*rand(1);
end