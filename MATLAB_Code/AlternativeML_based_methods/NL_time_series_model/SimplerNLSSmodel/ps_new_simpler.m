% Particle smoother based on Douc, Garivier, Moulines and Olsson (2010)

function g = ps_new_simpler(m,M,gPF, z)
  N     = size(gPF.xPResampled,1);            % Number of samples
  Qi    = inv(m.Q);
%  B     = 1/sqrt(((2*pi)^m.nx)*det(m.ss.Q));  % Upper bound for the pdf of the dynamics
%  const = 1/sqrt(((2*pi)^m.nx)*det(m.ss.Q));
  J     = zeros(N,M);              % The index used for the smoothed estimate
  xs    = zeros(1,N);
  xPart = zeros(N,M);
  nrR   = zeros(N,M);              % Keep track of how many rejections we get
  J(N,:) = floor(rand(1,M)*M)+1;   % Randomly select one of the equaly likely indecies
  u = z.u; % Input signal
  
  for t = N-1:-1:1
    for i=1:M
      notfound   = 1;
      nrRejected = 0;
      while notfound
        U      = rand(1);
        I      = ceil(rand(1)*M);     % Randomly select one of the equaly likely indecies
%        w      = gPF.xPResampled(t+1,J(t+1,i)) - m.ss.A*gPF.xPResampled(t,I);
        xt     = gPF.xPResampled(t,I);
        xt1    = gPF.xPResampled(t+1,J(t+1,i));
        %w      = xt1 - m.a*(xt/(1.0+xt^2)) - cos(1.2*t);
        %w      = xt1 - m.a*(xt/(1.0+xt^2)) - 8.*cos(1.2*t);
        %w      = xt1 - m.a*(xt/(1.0+xt^2)) - u(t);
        w      = xt1 - m.a*(xt/(1.0+(0.2.*xt).^2)) - u(t);
        target = exp(-0.5*w'*Qi*w);   % Evaluate the target density for the choosen index
        if U <= target
          J(t,i)   = I;
          notfound = 0;
        else
          nrRejected = nrRejected + 1;
        end
      end
      nrR(t,i) = nrRejected;
    end
  end

  for t=1:N
    xPart(t,:) = gPF.xPResampled(t,J(t,:));
    xs(t)      = mean(xPart(t,:));
  end
  g.xPart = xPart;      % Return the particles
  g.Xs    = xs;         % Return the smoothed states estimate
  g.J     = J;          % Return the resulting index
  g.nrR   = nrR;        % Return the number of rejections
end
