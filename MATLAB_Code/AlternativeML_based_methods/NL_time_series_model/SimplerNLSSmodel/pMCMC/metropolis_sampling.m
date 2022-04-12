function [x_current] = metropolis_sampling(x_prev, alpha)
    
    % Sample from the proposal distribution
    y_current = proposal_pdf_unif_sample(x_prev, alpha);
    
    rho = compute_acceptance_rate(x_prev, y_current, alpha);
    %disp(['Rho = ',num2str(rho)]);
    % Flip a fair coin and decide whether new sample is accepted or
    % rejected
    if rho > log(rand(1,1))
        x_current = y_current;
    else
        x_current = x_prev;
    end
        
end


    