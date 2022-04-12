function [G_discrete, x_model_discrete] = get_discr_model(u1, k, alpha, w0, xi, Ts)
    
    s = tf('s');
    
    % Build a ss model by first formulating the CT transfer function 
    G_continuous = (k*alpha*w0^2)/((s+alpha)*(s^2+2*xi*w0*s + w0^2));
    
    % convert the CT transfer function to DT by c2d using zero-order hold
    G_discrete = c2d(G_continuous, Ts, 'zoh'); 
    
    assert(isstable(G_discrete) == true, 'Resulting transfer function is unstable'); 
    %disp(['System is stable: ', num2str(isstable(G_discrete))]);
    % Then simulate the resulting DT model by lsim, store x (state
    % sequence). The variance var_e is just in the likelihood calc.
    % function
    x_model_discrete = abs(lsim(G_discrete, u1, (0:500-1).*Ts)); 
    
end