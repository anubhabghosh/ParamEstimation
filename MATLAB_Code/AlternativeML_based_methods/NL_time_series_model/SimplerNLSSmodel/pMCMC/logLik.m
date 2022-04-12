function value = logLik(theta,y,sigmaw,sigmav)
    
    value = zeros(size(theta));
    N = length(y);
    F = @(theta) toeplitz([0 1 (theta*ones(1,N-2)).^[1:N-2]], zeros(1,N));
    Sigma = @(theta) F(theta)*F(theta)'*sigmaw^2 + sigmav^2*eye(N);  % covariance of Y

    for idx = 1:length(theta)
       [U,S,V] = svd(Sigma(theta(idx)));
       value(idx) =  -0.5*y'*V*diag(1./diag(S))*U'*y - (N/2)*log(2*pi) - 0.5*sum(log(diag(S)));
    end

end
