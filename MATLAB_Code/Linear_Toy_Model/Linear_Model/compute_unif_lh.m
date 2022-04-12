%% Build a function that computes the likelihood using a Uniform distribution
% For scalars x, y:
% p(x) = 1/(b-a) if a <= x <= b, 0  elsewhere
% For multiple dimensions e.g. 2 dimensions, this means that:
% p(x, y) = 1/((b-a)*(d-c)) if a <= x <= b, c <= y <= d, 0 elsewhere

function [pU_x] = compute_unif_lh(x, limits)
% We assume 'x' is an Nx1-dimensional vector 
    [N, ~] = size(limits); % This contains the limits of each unknown parameter 
    pU_x = 1.0;
    for i=1:N
        a_i = limits(i, 1);
        b_i = limits(i, 2);
        assert(((b_i ~= 0) && (b_i > a_i)), 'Limits are improper, Exit!!');
        if (x(i) >= a_i && x(i) <= b_i)
            pU_x = pU_x .* (1.0/(b_i - a_i));
        else
            pU_x = 0.0;
        end
    end   
end