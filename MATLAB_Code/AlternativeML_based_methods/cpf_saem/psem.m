function [q,r] = psem(numIter, y, N, qinit, rinit, q0, r0, plotOn)
% Runs the Particle Smoothing EM algorithm with forward filtering/backward
% simulation particle (FFBSi) smoothing. We use the fast
% rejection-sampling-based FFBSi with early stopping, see
%
%   R. Douc, A. Garivier, E. Moulines and J. Olsson, "Sequential Monte
%   Carlo smoothing for general state space hidden Markov models" Annals of
%   Applied Probability, 2011, 21, 2109-2145.
%
% and
%  
%   E. Taghavi, F. Lindsten, L. Svensson and T. B. Schön, "Adaptive
%   stopping for fast particle smoothing" Proceedings of the 38th IEEE
%   International Conference on Acoustics, Speech and Signal Processing
%   (ICASSP), Vancouver, Canada, May 2013.
%
% The function returns the iterates for (q, r).

q = zeros(numIter,1);
r = zeros(numIter,1);
% Initialize the parameters
q(1) = qinit;
r(1) = rinit;
% Run a forward filter/backward simulator (FFBSi) particle smoother
bwdtraj = ffbsi(y, q(1), r(1), N);

if(plotOn)
    figure(2);clf(2);
    subplot(211);
    plot([1 numIter], q0*[1 1],'k-'); hold on;
    xlabel('Iteration');
    ylabel('q');
    subplot(212);
    plot([1 numIter],r0*[1 1],'k-'); hold on;
    xlabel('Iteration');
    ylabel('r');    
end

% Run identification loop
reverseStr = [];
for(k = 2:numIter)
    reverseStr = displayprogress(100*k/numIter, reverseStr);
    
    % Compute sufficient statistics
    S = compute_S(bwdtraj, y);
    % Maximize the intermediate quantity (here, the maximizing arguments
    % are simply given by the sufficient statistics)
    q(k) = S(1);
    r(k) = S(2);
    % Run CPF-AS
    bwdtraj = ffbsi(y, q(k), r(k), N);
    % Plot
    if(plotOn)
        subplot(211);
        plot(k, q(k),'r.');hold on;
        xlim([0 ceil(k/100)*100]);
        subplot(212);
        plot(k, r(k),'r.');hold on;
        xlim([0 ceil(k/100)*100]);
        drawnow;       
    end
end
end
%--------------------------------------------------------------------------
function x_bwd = ffbsi(y, q, r, N)
% Forward filter/backward simulator using rejection sampling and early
% stopping.
%
% Input:
%   y - measurements
%   q - process noise variance
%   r - measurement noise variance
%   N - number of particles/backward trajectories

T = length(y);
x = zeros(N, T); % Particles
w = zeros(N, T); % Weights
x(:,1) = 0; % Deterministic initial condition

% Forward filter
for(t = 1:T)
    if(t ~= 1)
        ind = resampling(w(:,t-1));
        xpred = f(x(:, t-1),t-1);
        x(:,t) = xpred(ind) + sqrt(q)*randn(N,1);
    end
    % Compute importance weights
    ypred = h(x(:,t));
    logweights = -1/(2*r)*(y(t) - ypred).^2; % (up to an additive constant)
    const = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    w(:,t) = weights/sum(weights); % Save the normalized weights
end
% Backward simulator
x_bwd = zeros(N,T);
ind = resampling(w(:,T));
x_bwd(:,T) = x(ind,T);
for(t = (T-1) : (-1) : 1)
   % Using fast RS-FFBSi with early stopping
    L = 1:N; % The list of particle indices that still need assignment       
    bins = cumsum(w(:,t)); % Precomputation
    counter = 0;
    while(~isempty(L) && counter < N/5) % The constant N/5 is quite arbitrary, but it only affects the computational time (i.e. not the result)
        n = length(L);
        [~, I] = histc(rand(n,1), [0 ; bins]);
        U = rand(n, 1);
        xt1 = x_bwd(L,t+1); % x^k_{t+1} for k \in L
        fxt = f(x(I,t),t); % f(x^i_t) for i \in I (candidates)
        p = exp(-1/(2*q)*(xt1 - fxt).^2);
        accept = (U <= p);
        x_bwd(L(accept),t) = x(I(accept),t);
        L = L(~accept);
        counter = counter+1;
    end
    if(~isempty(L))
        for(j = L)
            fxt = f(x(:,t),t);
            logp = -1/(2*q)*(x_bwd(j,t+1) - fxt).^2;
            const = max(logp); % Subtract the maximum value for numerical stability
            w_sm = exp(logp-const).*w(:,t);
            w_sm = w_sm/sum(w_sm);
    
            ind = find(rand(1) < cumsum(w_sm),1,'first');
            x_bwd(j,t) = x(ind,t);
        end
    end
end
end
%-------------------------------------------------------------------
function S = compute_S(X,y)
% Compute the sufficient statistic for PSEM
[N,T] = size(X);
fX = f(X(:,1:T-1), repmat((1:T-1),[N,1]));
S1 = mean(mean((X(:,2:T)-fX).^2)); % Mean over t and j

hX = h(X);
S2 = mean(mean((repmat(y,[N,1])-hX).^2)); % Mean over t and j

S = [S1 ; S2];
end
%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end
