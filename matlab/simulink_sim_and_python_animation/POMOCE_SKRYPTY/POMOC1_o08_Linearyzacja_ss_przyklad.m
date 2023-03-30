%{
    Przykład wykorzystania funkcji linearyzacja_ss.m
%}

clc;

syms x1 x2 y1 y2 u1 u2

%% Wektory stanu, wyjscia, sterowania, funkcje
x_vec = [x1; x2];
y_vec = [y1; y2];
u_vec = [u1; u2];

args_vec = [x_vec ; u_vec];

% Funkcja wektorowa równania stanu
f1 = x1^2+x2+3*u1^2+25*u2-y1+y2;    % f1(x, u)
f2 = 2*x1+4*x2+u1-u2-y1+27*u2;      % f2(x, u)
f_vec = [f1; f2];

% Funkcja wektorowa równania wyjścia
h1 = 0;
h2 = 0;
h_vec = [h1; h2];

% Punkt pracy
x1_pp=1; x2_pp=1; u1_pp=1; u2_pp=1;
S0 = [x1_pp; x2_pp; u1_pp; u2_pp];


%% Wywołanie funkcji
[A, B, C, D] = linearyzacja_ss(f_vec, h_vec, x_vec, u_vec, S0);











