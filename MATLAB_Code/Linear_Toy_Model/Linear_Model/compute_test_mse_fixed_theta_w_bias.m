function [mse_pred_actual, mse_pred_asymp, predicted_theta, asymp_theta, ...
    actual_theta, A_analytical, b_analytical, A_asymptotic, b_asymptotic] ...
    = compute_test_mse_fixed_theta_w_bias(d, M, p, N, Phi, Y, actual_theta, lambda_e)
    
    disp(['P = ', num2str(p), ' M = ', num2str(M)]);
    % least squares
    %A_ls = inv(Phi'*Phi)*Phi';
    %theta_ls = A_ls*Y;
    
    % new estimator
    %theta_i = 2*rand(length(theta),p);
    disp('Creating training data for estimation...');
    %theta_min = 0;
    %theta_max = 2;
    
    %theta_i = 2*rand(d, p); % Uniform prior
    var_theta = [1/3 0; 0 1/3;];
    mu_theta = [1; 1;];
    
    theta_i = 1.0 + chol(var_theta)*randn(d, p); % Gaussian prior
    
    %figure;scatter(theta_i(1, :), theta_i(2, :));
    
    % Obtain asymptotic estimate
    disp('Computing theta_hat_asymptotic');
    d = 2;
    
    
    [asymp_theta, A_asymptotic, b_asymptotic] = compute_asymptotic_theta_estimate(d, p, N, theta_i, Phi, Y, mu_theta, var_theta, lambda_e);
    
    %mu_theta  = (theta_max+theta_min)/2;
    %var_theta = (1/12)*(theta_max - theta_min)^2;
    %Phi1 = Phi(:,1);
    %Phi2 = Phi(:,2);
    
    %disp('Computing theta_hat_asymptotic numerically');
    %lambda_e = 0.3^2; % 1.0
    %[asymp_theta_analytical, asymp_theta_num] = compute_asymptotic_theta_numerically(d, lambda_e, mu_theta, var_theta, Phi, Phi1, Phi2, Y);
    
    % Obtain analytical estimate (solution of the optimization problem)
    Ys = zeros(N,M,p);
    for i = 1:p
        Ys(:,:,i) =  Phi*theta_i(:,i) + sqrt(lambda_e).*randn(N,M);
    end
    Ys = [Ys; ones(1, M, p)]; % Add a bias term of 1 at the end
    disp('Evaluating closed form expression...');
    % evaluating closed form expression
    M1 = zeros(2*(N+1),1);
    M2 = zeros(2*(N+1),2*(N+1));
    
    for i=1:p
        for m =1:M
            M1 = M1 + kron(Ys(:,m,i)',eye(2))'*theta_i(:,i);
        end
    end
    
    for i=1:p
        for m =1:M
            M2 = M2 + kron(Ys(:,m,i)',eye(2))'*kron(Ys(:,m,i)',eye(2));
        end
    end
    
    disp('Reshaping matrix');
    Ab_analytical = reshape(M2\M1,[2,(N+1)]);
    A_analytical = Ab_analytical(:,1:end-1);
    b_analytical = Ab_analytical(:,end);
    %disp(size(A_analytical));
    %disp(size(Y));
    
    disp('Computing theta_hat_analytical');
    theta_hat_analytical = A_analytical*Y' + b_analytical;
    
    % alpha_hat = fminunc(@(alpha) criterion(Ys, theta_i, alpha,p,M), ones(2*N,1) );
    % theta_hat = reshape(alpha_hat,[2,N])*Y
    %
    % function cost = criterion(Ys, theta_i, alpha_hat, p, M)
    % cost = 0;
    % for i = 1:p
    %     for m=1:M
    %         cost =   cost + norm(theta_i(:,i) - kron(Ys(:,m,i)',eye(2))*alpha_hat)^2;
    %     end
    % end
    % end
    
    predicted_theta = theta_hat_analytical';
    mse_pred_actual = immse(predicted_theta, actual_theta);
    mse_pred_asymp = immse(predicted_theta, asymp_theta);
    %mse_asymp_num = immse(asymp_theta_num, asymp_theta_analytical);
    
end