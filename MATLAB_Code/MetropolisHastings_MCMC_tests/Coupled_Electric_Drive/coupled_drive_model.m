% model obtained in Coupled Electric Drives Data Set and Reference Models
close all

% load data
load('DATAPRBS.mat')
Ts = 20e-3;
figure(1)
subplot(2,1,2)
plot((1:500).*Ts, u1)
grid on
subplot(2,1,1)
plot((1:500).*Ts, z1)
grid on
hold all

%% 
s = tf('s');

% possible model  (table 2 in section 3.4, second row)
a =  5163;
b = -19.93;
c = -509.8;
d = -2835;
G = a/(s^3 -b*s^2 - c*s -d);


k = -a/d;             % a/d;
alpha = -6.73507; % obtained by solving qubic equation(roots([1 -b -c -d])) 
                  % alpha^3 + b alpha^2 + c alpha +d = 0  
                  % take real solution
alpha = -alpha;
w0 = sqrt(-d/alpha);%w0 = sqrt(d/alpha);     %
xi = (-b - alpha)/(2*w0);               %(b-alpha)/(2*w0);
Gtilde = (k*alpha*w0^2)/((s+alpha)*(s^2+2*xi*w0*s + w0^2));  % same as G
disp(["Is constructed transfer function stable?", num2str(isstable(Gtilde))]);

figure(2)
bode(G)
y_model = abs(lsim(G,u1,(0:500-1).*Ts));
figure(1)
subplot(2,1,1)
plot((1:500).*Ts, y_model)

% simulation with euler discretization
A = [0 1 0;
     0 0 1;
     d c b];
B = [0; 0; a;];
C = [1 0 0];

A_euler = eye(3) + Ts*A;
B_euler = Ts*B;
C_euler = C;

%construct Euler discretized state space 
model_euler = ss(A_euler, B_euler, C_euler, 0, 1);
y_model_euler = abs(lsim(model_euler,u1));
figure(1)
subplot(2,1,1)
plot((1:500).*Ts, y_model_euler)
legend('measured output', 'model (CT)', 'model (Euler)')