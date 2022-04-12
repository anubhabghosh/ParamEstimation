function [llh_p_aliter] = get_mvn_llh(X, mu, Var)
    
    % Get the approx log-likelihood for each sample, assuming that we have 
    % a multivariate Gaussian distribution
    N = size(X, 1);
    %for i=1:N
        %llh_p_aliter(i, :) = 1.0 ./ sqrt((2*pi).^2 * det(Sigma)) * exp(-0.5 .* ((X(i, :) - mu) \ (Sigma)) * transpose(X(i, :) - mu));
    llh_p_aliter =  -(N / 2.0).*log(2 * pi * Var) - ((1.0 ./ (2.*Var)).*norm(X - mu, 2).^2);
    %end

end

