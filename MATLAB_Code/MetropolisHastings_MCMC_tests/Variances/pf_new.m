% Particle filter

function [g, approxLogLik] = pf_new(m,M,z)
  y           = z.y;
  N           = length(y);       % Number of samples
  xf          = zeros(1,size(y,2));
  xPWeighted  = zeros(N,M);      % Make some room
  xPResampled = zeros(N,M);      % Make some room
  wStore      = zeros(N,M);      % Make some room
  vStore      = zeros(N,M);      % Make some room


  % 1. Initialize particles
  if m.P1==0
    x = repmat(m.X1,1,M);    % Initial state fully known
  else
    x = repmat(m.X1,1,M) + sqrt(m.P1)*randn(1,M);
  end
  
  approxLogLik = 0;
  
  for t=1:N
    %e = repmat(y(t),1,M) - x;  % output equation
    e = repmat(y(t),1,M) - 0.05*x.^2;
    %e = repmat(y(t),1,M) - 0.05*x;
    v = -0.5*log(2*pi*m.R)-(0.5*(e.*(m.R\e)));  % 2. Evaluate the log-likelihood
    maxWeight = max(v);
    v = exp(v - maxWeight);
    w = v/sum(v);
    xPWeighted(t,:) = x;
    wStore(t,:)     = w; % normalized weights > their sum is equal to 1
    vStore(t,:)     = v; % unnormalized shifted log weights
    xf(t) = sum(w.*x);             % Compute state estimate
    index = sysresample(w);        % 3. Resample
    x     = x(index);
    xPResampled(t,:) = x;
    %x = m.a*x + sqrt(m.Q)*randn(1,M);
    x = 0.5*x + 25.0*(x./(1+x.^2)) + 8.0*cos(1.2*t) + sqrt(m.Q)*randn(1,M); % 4. Predict the particles (state equation)
    %x = 0.5*x + 8.0*cos(1.2*t) + sqrt(m.Q)*randn(1,M); % 4. Predict the particles (state equation)
    
    % compute approximation of the logLikelihood
    approxLogLik = approxLogLik + (maxWeight + log(sum(v)) - log(M)); 
    
  end
  g.Xf          = xf;
  g.xPWeighted  = xPWeighted;
  g.xPResampled = xPResampled;
  g.w           = wStore;
  g.v           = vStore; % unnormalized shifted log weights
  
  
end
