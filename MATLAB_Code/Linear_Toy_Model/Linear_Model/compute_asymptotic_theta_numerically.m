function [asymp_theta, Gaussian_prior] = compute_asymptotic_theta_numerically(d, lambdae, mu_theta, var_theta, Phi, Phi1, Phi2, Y)
    
    N = size(Y, 2);
    asymp_theta = var_theta*Phi'/(Phi*var_theta*Phi'+ lambdae*eye(N))*Y'   +  (eye(d) - (var_theta*Phi'/(Phi*var_theta*Phi'+ lambdae*eye(N))*Phi))*[mu_theta; mu_theta;];
    Gaussian_prior = zeros(d, size(Y, 1));
    
    for i=1:size(Gaussian_prior, 2)
        
        Yi = Y(i, :)';
        integrand_normalization_gaussian = @(x,y) posterior_gaussian(x,y, N, Yi, Phi1, Phi2, lambdae, mu_theta, var_theta);
        %integrand_normalization_uniform  = @(x,y) posterior_uniform(x,y, N,Y,Phi1, Phi2,lambdae);

        normalization_gaussian = integral2(integrand_normalization_gaussian,-inf,inf,-inf,inf,'AbsTol',1e-100,'method','iterated');
        %normalization_uniform  = integral2(integrand_normalization_uniform,theta_min,theta_max,theta_min,theta_max,'AbsTol',1e-100,'method','iterated');

        integrand_gaussian_case1 = @(x,y) x.*posterior_gaussian(x,y, N,Yi,Phi1, Phi2,lambdae, mu_theta, var_theta)/normalization_gaussian;
        %integrand_uniform_case1  = @(x,y) x.*posterior_uniform(x,y, N,Y,Phi1, Phi2,lambdae)/normalization_uniform;
        integrand_gaussian_case2 = @(x,y) y.*posterior_gaussian(x,y, N,Yi,Phi1, Phi2,lambdae, mu_theta, var_theta)/normalization_gaussian;
        %integrand_uniform_case2  = @(x,y) y.*posterior_uniform(x,y, N,Y,Phi1, Phi2,lambdae)/normalization_uniform;

        posterior_mean_Gaussian_prior_theta1 = integral2(integrand_gaussian_case1,-inf,inf,-inf,inf,'AbsTol',1e-100,'method','iterated');
        posterior_mean_Gaussian_prior_theta2 = integral2(integrand_gaussian_case2,-inf,inf,-inf,inf,'AbsTol',1e-100,'method','iterated');

        %posterior_mean_uniform_prior_theta1  = integral2(integrand_uniform_case1,theta_min,theta_max,theta_min,theta_max,'AbsTol',1e-100,'method','iterated');
        %posterior_mean_uniform_prior_theta2  = integral2(integrand_uniform_case2,theta_min,theta_max,theta_min,theta_max,'AbsTol',1e-100,'method','iterated');

        Gaussian_prior(:, i) = [posterior_mean_Gaussian_prior_theta1; posterior_mean_Gaussian_prior_theta2;];
        %Uniform_prior = [posterior_mean_uniform_prior_theta1; posterior_mean_uniform_prior_theta2;];
    
    end
    
    disp(['error Gaussian ', num2str(immse(Gaussian_prior-Analytical))])
    
end

%Gaussian case
function value = posterior_gaussian(theta1, theta2, N,Y,Phi1, Phi2, lambdae, mu, var)
    value =  ((2*pi*lambdae)^(-N/2))*exp(-(0.5/lambdae)*(Y'*Y - (2*Phi1'*Y).*theta1 - (2*Phi2'*Y).*theta2 + 2*(Phi1'*Phi2)*(theta1.*theta2) + (Phi1'*Phi1)*theta1.^2 + (Phi2'*Phi2)*theta2.^2  )) .*...
        (1/(2*var*pi)).*exp(-(1/(2*var))*((theta1-mu).^2 +(theta2-mu).^2));
end

%Uniform case
%function value = posterior_uniform(theta1, theta2, N,Y,Phi1, Phi2,lambdae)
%value = ((2*pi*lambdae)^(-N/2))*exp(-(0.5/lambdae)*(Y'*Y - (2*Phi1'*Y).*theta1 - (2*Phi2'*Y).*theta2 + 2*(Phi1'*Phi2)*(theta1.*theta2) + (Phi1'*Phi1)*theta1.^2 + (Phi2'*Phi2)*theta2.^2  ));
%end