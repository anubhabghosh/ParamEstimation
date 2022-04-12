function [asymp_theta, A_asymptotic, b_asymptotic] = compute_asymptotic_theta_estimate(d, p, N, theta_i, Phi, Y, mu_theta, var_theta, lambda_e)

    % Obtain the asymptotic value of A
    % Calculate the expected value 
    % R_theta = lim p \to \infty (1/p) \sum_{i=1}^{p} R_{\theta} \Phi^{\top} \left(\Phi R_{\theta} \Phi^{\top} + \lambda_e I_{d} \right)^{-1}
    %R_theta = zeros(d, d);
    %for p_idx=1:p
    %    R_theta = R_theta + (theta_i(:, p_idx) * theta_i(:, p_idx)');
    %end
    %R_theta = (1.0 / p) * R_theta; % Normalization
    
    R_theta = var_theta;
    A_tmp = Phi * (R_theta * Phi') + lambda_e * eye(N);
    A_asymptotic = (R_theta * Phi') / A_tmp;
    b_asymptotic = (eye(d) - A_asymptotic * Phi)*mean(theta_i, 2);
    %b_asymptotic = (eye(d) - A_asymptotic * Phi)*mu_theta;
    %disp(mu_theta - mean(theta_i, 2));
    
    %asymp_theta = (A_asymptotic * Y')';
    asymp_theta = (A_asymptotic * Y' + b_asymptotic)';
    
end