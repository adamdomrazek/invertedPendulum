%{
    Linearyzacja modelu wahadła, wykorzystanie funkcji 'zz_linearyzacja_ss_4na4.m'
%}

clc;

syms x1 x2 x3 x4 u Mt m g l Jt b gamma y1 y2

%% Wektory stanu, wyjscia, sterowania, funkcje
x_vec = [x1; x2; x3; x4];
y_vec = [y1; y2];
u_vec = u;

args_vec = [x_vec ; u_vec];

% Funkcje równań stanu
f1 = x3;
f2 = x4; 
f3 = ((m^2 * l^2 * g / Jt) * sin(x2) * cos(x2) + m*l*gamma/Jt * x4 * cos(x2) + m*l*x4^2*sin(x2) - b*x3 + u ) / ( Mt - m^2*l^2/Jt * cos(x2) ) ;
f4 = ( (Mt*m*g*l/Jt)*sin(x2) - (Mt*gamma/Jt)*x4 +m*l*b*x3*cos(x2) - m^2*l^2*x4^2*sin(x2)*cos(x2) - m*l*cos(x2)*u) / ( Mt - m^2*l^2/Jt * cos(x2) ) ;
f_vec = [f1; f2; f3; f4];

% Funkcje pomiarów
h1 = x1;
h2 = x2;
h3 = x3;
h4 = x4;
h_vec = [h1; h2; h3; h4];

% Punkt pracy
% [x1, x2, x3, x4] = [x, theta, Dx, Dtheta]
x1_pp=0; x2_pp=pi; x3_pp=0; x4_pp=0; u_pp=0;
S0 = [x1_pp; x2_pp; x3_pp; x4_pp; u_pp];


%% Wywołanie funkcji
[A, B, C, D] = zz_linearyzacja_ss_4na4(f_vec, h_vec, x_vec, u_vec, S0);

%% Podstawienie danych do A
A = subs(A, g, 9.81);
A = subs(A, l, 1);
A = subs(A, m, 1);
A = subs(A, Mt, 2);
A = subs(A, b, 0.2);
A = subs(A, Jt, 3);
A = subs(A, gamma, 0.1);
A = double(A);

B = subs(B, l, 1);
B = subs(B, m, 1);
B = subs(B, Mt, 2);
B = subs(B, Jt, 3);
B = double(B);

C = double(C);
D = double(D);

%% sys- cos nie dziala
sys = ss(A, B, C, D);
step(sys)







