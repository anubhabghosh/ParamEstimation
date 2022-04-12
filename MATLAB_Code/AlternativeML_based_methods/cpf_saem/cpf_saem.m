function [q,r] = cpf_saem(numIter, y, N, gamma, qinit, rinit, q0, r0, plotOn)
% Runs the CPF-SAEM algorithm,
%
%   F. Lindsten, "An efficient stochastic approximation EM algorithm
%   using conditional particle filters", Proceedings of the 38th
%   International Conference on Acoustics, Speech, and Signal Processing
%   (ICASSP), Vancouver, Canada, May 2013.
%
% The function returns the iterates for (q, r).

T = length(y);
q = zeros(numIter,1);
r = zeros(numIter,1);
X = zeros(numIter,T);
% Initialize the parameters
q(1) = qinit;
r(1) = rinit;
% Initialize the state by running a PF
[particles, w] = cpf_as(y, q(1), r(1), N, X);
% Draw J
J = find(rand(1) < cumsum(w(:,T)),1,'first');
X(1,:) = particles(J,:);
S = [0 ; 0]; % Sufficient statistics for (q,r)

if(plotOn)
    figure(1);clf(1);
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
    
    % Update sufficient statistics
    S = update_S(S, particles, w(:,T), gamma(k), y);
    % Maximize the intermediate quantity (here, the maximizing arguments
    % are simply given by the sufficient statistics)
    q(k) = S(1);
    r(k) = S(2);
    % Run CPF-AS
    [particles, w] = cpf_as(y, q(k), r(k), N, X(k-1,:));
    % Draw J (extract a particle trajectory)
    J = find(rand(1) < cumsum(w(:,T)),1,'first');   
    X(k,:) = particles(J,:);    
    % Plot
    if(plotOn)
        subplot(211);
        plot(k, q(k),'b.');hold on;
        xlim([0 ceil(k/100)*100]);
        subplot(212);
        plot(k, r(k),'b.');hold on;
        xlim([0 ceil(k/100)*100]);
        drawnow;       
    end
end
end
%--------------------------------------------------------------------------
function [x,w] = cpf_as(y, q, r, N, X)
% Conditional particle filter with ancestor sampling
% Input:
%   y - measurements
%   q - process noise variance
%   r - measurement noise variance
%   N - number of particles
%   X - conditioned particles - if not provided, un unconditional PF is run

conditioning = (nargin > 4);
T = length(y);
x = zeros(N, T); % Particles
a = zeros(N, T); % Ancestor indices
w = zeros(N, T); % Weights
x(:,1) = 0; % Deterministic initial condition
x(N,1) = X(1); % Set the 1st particle according to the conditioning

for(t = 1:T)
    if(t ~= 1)
        ind = resampling(w(:,t-1));
        ind = ind(randperm(N));
        xpred = f(x(:, t-1),t-1);
        x(:,t) = xpred(ind) + sqrt(q)*randn(N,1);
        if(conditioning)
            x(N,t) = X(t); % Set the N:th particle according to the conditioning
            % Ancestor sampling
            m = exp(-1/(2*q)*(X(t)-xpred).^2);
            w_as = w(:,t-1).*m;
            w_as = w_as/sum(w_as);
            ind(N) = find(rand(1) < cumsum(w_as),1,'first');
        end
        % Store the ancestor indices
        a(:,t) = ind;
    end
    % Compute importance weights
    ypred = h(x(:,t));
    logweights = -1/(2*r)*(y(t) - ypred).^2; % (up to an additive constant)
    const = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    w(:,t) = weights/sum(weights); % Save the normalized weights
end

% Generate the trajectories from ancestor indices
ind = a(:,T);
for(t = T-1:-1:1)
    x(:,t) = x(ind,t);
    ind = a(ind,t);
end
end
%-------------------------------------------------------------------
function S = update_S(S,X,w,gamma,y)
% Update the sufficient statistic for SAEM
[N,T] = size(X);
fX = f(X(:,1:T-1), repmat((1:T-1),[N,1]));
S1 = mean((X(:,2:T)-fX).^2,2); % Mean over t
S1 = S1'*w;

hX = h(X);
S2 = mean((repmat(y,[N,1])-hX).^2,2); % Mean over t
S2 = S2'*w;

S = (1-gamma)*S + gamma*[S1 ; S2];
end
%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end
