function [mse, predicted_theta, actual_theta] = compute_test_mse_sampled_theta(d, M, p, N, K_test)
    
    % data set 
    %theta = [1; 1;];
    actual_theta = 0.5 + (1.5-0.5)*rand(K_test, d, 1);
    %actual_theta = 2.*rand(K_test, d, 1);
    
    Y = zeros(K_test, N);
    for i=1:length(actual_theta)
        E = randn(N,1);
        U = rand(N,1);
        Phi = [U [0; U(1:N-1);] ];
        Y(i, :) = Phi*actual_theta(i, :)' + E;
    end
    
    % least squares
    %A_ls = inv(Phi'*Phi)*Phi';
    %theta_ls = A_ls*Y;
    
    % new estimator
    %theta_i = 2*rand(length(theta),p);
    theta_i = 2*rand(d, p);
    Ys = zeros(N,M,p);
    for i = 1:p
        Ys(:,:,i) =  Phi*theta_i(:,i) + randn(N,M);
    end
    
    % evaluating closed form expression
    M1 = zeros(2*N,1);
    M2 = zeros(2*N,2*N);
    
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
    
    A_analytical = reshape(M2\M1,[2,N]);
    %disp(size(A_analytical));
    %disp(size(Y));
    theta_hat_analytical = A_analytical*Y';
    
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
    mse = immse(predicted_theta, actual_theta);
end