function [k_llim, alpha_llim, w0_llim, xi_llim] = get_unif_lower_limit(prbs_struct, percent_)
    
    k_llim = (1 - percent_) .* prbs_struct.k_bf;
    alpha_llim = (1 - percent_) .* prbs_struct.alpha_bf;
    w0_llim = (1 - percent_) .* prbs_struct.omega0_bf;
    xi_llim = (1 - percent_) .* prbs_struct.xi_bf;
    
end