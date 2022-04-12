% Particle filter

function g = pf_new_simpler(m,M,z)
  y           = z.y;
  N           = size(y,2);       % Number of samples
  xf          = zeros(1,size(y,2));
  xPWeighted  = zeros(N,M);      % Make some room
  xPResampled = zeros(N,M);      % Make some room
  wStore      = zeros(N,M);      % Make some room
  
  u = z.u;
  % 1. Initialize particles
  if m.P1==0
    x = repmat(m.X1,1,M);    % Initial state fully known
  else
    x = repmat(m.X1,1,M) + sqrt(m.P1)*randn(1,M);
  end
  for t=1:N
    e = repmat(y(t),1,M) - m.b*x.^2;
    %e = repmat(y(t),1,M) - m.b*x;
    w = exp(-(1/2)*(e.*(m.R\e)));  % 2. Evaluate the likelihood
    w = w/sum(w);
    xPWeighted(t,:) = x;
    wStore(t,:)     = w;
    xf(t) = sum(w.*x);             % Compute state estimate
    index = sysresample(w);        % 3. Resample
    x     = x(index);
    xPResampled(t,:) = x;
    %x = m.a*(x./(1.0+x.^2)) + cos(1.2*t) + sqrt(m.Q)*randn(1,M); % 4. Predict the particles
    %x = m.a*(x./(1.0+x.^2)) + 8.*cos(1.2*t) + sqrt(m.Q)*randn(1,M); % 4. Predict the particles
    %x = m.a*(x./(1.0+x.^2)) + u(t) + sqrt(m.Q)*randn(1,M); % 4. Predict the particles
    x = m.a*(x./(1.0+ (0.2.*x).^2)) + u(t) + sqrt(m.Q)*randn(1,M); % 4. Predict the particles
  end
  g.Xf          = xf;
  g.xPWeighted  = xPWeighted;
  g.xPResampled = xPResampled;
  g.w           = wStore;
end
