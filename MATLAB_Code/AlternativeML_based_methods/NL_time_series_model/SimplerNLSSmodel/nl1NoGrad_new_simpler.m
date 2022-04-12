% Used in solving the nonlinear identification problem considered in 
% Section 10.2 in
%
% Thomas B. Schön, Adrian Wills and Brett Ninness. System Identification 
% of Nonlinear State-Space Models. Automatica, 47(1):39-49, January 2011. 
%
% theta  = current value for the parameters.
% xPart  = particles to be used in order to form the Q function.
% y      = measurements to be used in order to form the Q function.
% 
% Qfun   = the value of the Q function at the point theta
%

function Qfun = nl1NoGrad_new_simpler(theta,xPart,z)
  
  y = z.y;
  u = z.u;
  a = theta(1);   
  b = theta(2);   
  Q = theta(3);   
  R = theta(4);

  N = size(xPart,1);   % Number of samples
  M = size(xPart,2);   % Number of particles 

  Qfun1 = 0;
  Qfun2 = 0;
  for t=1:N-1
    %Qfun1tmp = sum((xPart(t+1,:) - a*xPart(t,:)./(1.0+(xPart(t,:)).^2) - cos(1.2*t)).^2);
    %Qfun1tmp = sum((xPart(t+1,:) - a*xPart(t,:)./(1.0+(xPart(t,:)).^2) - 8.*cos(1.2*t)).^2);
    %Qfun1tmp = sum((xPart(t+1,:) - a*xPart(t,:)./(1.0+(xPart(t,:)).^2) - u(t)).^2);
    Qfun1tmp = sum((xPart(t+1,:) - a*xPart(t,:)./(1.0+(0.2.*(xPart(t,:))).^2) - u(t)).^2);
    Qfun1    = Qfun1 + (1/(2*M*Q*Q))*Qfun1tmp;
  end
  for t=1:N
    Qfun2tmp = sum((y(t) - b*xPart(t,:).^2).^2);
    %Qfun2tmp = sum((y(t) - b*xPart(t,:)).^2);
    Qfun2    = Qfun2 + (1/(2*M*R*R))*Qfun2tmp;
  end
  Qfun = -((N-1)/2)*log(Q*Q) - Qfun1 - (N/2)*log(R*R) - Qfun2;
  Qfun = -Qfun;       % We want to maximize Q, but fminunc solves a minimization problem
end