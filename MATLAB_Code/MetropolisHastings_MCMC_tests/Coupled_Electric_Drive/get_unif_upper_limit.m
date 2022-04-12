function [k_ulim, alpha_ulim, w0_ulim, xi_ulim] = get_unif_upper_limit(prbs_struct, percent_)
    
    k_ulim = (1 + percent_) .* prbs_struct.k_bf;
    alpha_ulim = (1 + percent_) .* prbs_struct.alpha_bf;
    w0_ulim = (1 + percent_) .* prbs_struct.omega0_bf;
    xi_ulim = (1 + percent_) .* prbs_struct.xi_bf;
    
end