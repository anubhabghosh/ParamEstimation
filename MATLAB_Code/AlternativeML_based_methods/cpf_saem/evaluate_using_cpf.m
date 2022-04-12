% This script contains the function file that calls the Maximum likelihood
% based particle filter and particle smoother for parameter estimation

function [q1, r1] = evaluate_using_cpf(y, N, M)

    %% CPF-SAEM ===============================================================
    
    % Set up some parameters
    N1 = 15;                % Number of particles used in CPF-SAEM
    T = N;                % Length of data record
    numIter = 500;          % Number of iterations in EM algorithms
    
    plotOn  = 0;            % Plot intermediate results (don't plot intermeditae results)
    kappa = 1;              % Constant used to compute SA step length (see below)
    
    % Generate data
    q0 = 1.0;  % True process noise variance (changing this 1, originally 0.1)
    r0 = 0.1; % True measurement noise variance
    
    %[x0,y0] = generate_data(T,q0,r0);
    
    % Initialization for the parameters (default in the code)
    %qinit = 1;
    %rinit = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Lower and Upper limits of theta_6(q) and theta_7(r) (for random choice)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    l_q = 0.1;
    u_q = 1.5;
    l_r = 1e-6;
    u_r = 1;
    
    %qinit = l_q + (u_q - l_q).*rand(1,1);
    rinit = l_r + (u_r - l_r).*rand(1,1);
    qinit = 1.0;
    %rinit = 1.0;
    
    % SA step length
    gamma = zeros(1,numIter);
    gamma(1:2) = 1;
    gamma(3:99) = 0.98;
    gamma(100:end) = 0.98*(((0:numIter-100)+kappa)/kappa).^(-0.7);
    
    % Run the algorithms
    fprintf('Running CPF-SAEM (N=%i). Q_init: %d, R_init: %f, Progress: %f ', N1,qinit,rinit); 
    tic;
    [q1,r1] = cpf_saem(numIter, y, N1, gamma, qinit, rinit, q0, r0, plotOn);
    timeelapsed = toc;
    fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed);

end
