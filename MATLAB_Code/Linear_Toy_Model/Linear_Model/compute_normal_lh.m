%% Build a function that computes the likelihood using a Gaussian distribution
% For vector x:
% p(x) = (det(2.*pi.*Sigma))^(-1).*(exp(-0.5.*(x - mu)'*(Sigma\(x - mu)))) 

function [fn_pN_x, pN_x] = compute_normal_lh(x, mu, Cov)
    
    N_data = size(x, 1);
    fn_pN_x = @(x, mu, Cov) (det(2.*pi.*Cov))^(-1).*(exp(-0.5.*(x - mu)'*(Cov\(x - mu))));
    if N_data == 1
        pN_x = fn_pN_x(x, mu, Cov);
    elseif N_data > 1
        pN_x = zeros(N_data, 1);
        for i=1:N_data
            pN_x(i) = fn_pN_x(x(i, :)', mu', Cov);
        end
    end
end