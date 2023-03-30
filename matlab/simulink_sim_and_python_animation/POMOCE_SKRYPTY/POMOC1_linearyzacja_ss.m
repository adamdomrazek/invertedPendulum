function [A,B,C,D] = linearyzacja_ss(f_vec, h_vec, x_vec, u_vec, S0)
%{
    Argumenty: 
        f_vec
        h_vec
        x_vec
        u_vec
        S0      -   Punkt pracy

    Linearyzacja tak jak w pliku "o02_syms_linearyzajca_Rstanu.mlx"
    Dla systemu:
        Dx(t) = f(x, u)
         y(t) = h(x, u)

    Funkcja przyjumje jako argumenty symboliczną funkcję f(...) i h(...)
    oraz wektory ich argumentów. Zwraca maacierze zlinearyzowanego systemu.
%}

%% Gradienty transponowane do wpakowania w macierz
grad_f1_x = gradient(f_vec(1), [x_vec]).';   % transpozycja A.' albo tanspose(A)
grad_f2_x = gradient(f_vec(2), [x_vec]).';   % transpozycja A.' albo tanspose(A)

grad_f1_u = gradient(f_vec(1), [u_vec]).';
grad_f2_u = gradient(f_vec(2), [u_vec]).';

grad_h1_x = gradient(h_vec(1), [x_vec]).';
grad_h2_x = gradient(h_vec(2), [x_vec]).';

grad_h1_u = gradient(h_vec(1), [u_vec]).';
grad_h2_u = gradient(h_vec(2), [u_vec]).';

%% Macierz gradientów f_vec po x_vec symboliczna 
nabla_fVec_x = [grad_f1_x; grad_f2_x];

%% Macierz gradientów f_vec po u_vec symboliczna 
nabla_fVec_u = [grad_f1_u; grad_f2_u];

%% Macierz gradientów h_vec po x_vec symboliczna 
nabla_hVec_x = [grad_h1_x; grad_h2_x];

%% Macierz gradientów h_vec po u_vec symboliczna 
nabla_hVec_u = [grad_h1_u; grad_h2_u];

args_vec = [x_vec; u_vec];

A = subs(nabla_fVec_x, args_vec, S0);
B = subs(nabla_fVec_u, args_vec, S0);
C = subs(nabla_hVec_x, args_vec, S0);
D = subs(nabla_hVec_u, args_vec, S0);

end

