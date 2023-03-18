'''
    model derivation all in one
'''
#%% IMPORTS
from IPython.display import display as disp
from IPython.display import Math as math
from IPython.display import Markdown as md

import numpy as np
from sympy import symbols, cos, sin
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import dill

# from sympy.physics.mechanics import init_vprinting
# from sympy.physics.mechanics import dynamicsymbols 
# from sympy import print_latex

#%% SYMBOLE & FUNKCJE (SYMPY)
# Constants & time
m_r, m_p, m_c, M, gamma, b, g, J_cm, L_c, L_p, L = symbols('m_r m_p m_c M gamma b g J_{cm} L_c L_p, L')
t = symbols('t')

# Functions
x, y, u, xcm, ycm = symbols('x y u x_{cm} y_{cm}', cls=smp.Function)
theta = symbols('theta', cls=smp.Function)

# Disturbances
D = smp.symbols('D', cls=smp.Symbol)
alpha = smp.symbols('alpha', cls=smp.Function)
alpha = alpha(t)

x = x(t)
y = y(t)
u = u(t)
xcm = xcm(t)
ycm = ycm(t)
theta = theta(t)

Rx, Ry = symbols('R_x R_y')

# Derivatives
Dx = x.diff(t, 1)
DDx = x.diff(t, 2)

#%% Trolley forces, friction model
F_f = b*Dx
lhs = M*DDx
rhs = u - F_f - Rx
f_trolley_eq = smp.Eq(lhs=lhs, rhs=rhs)
print(f_trolley_eq)

#%% L: distance from axis of rotation to center of mass of pendulum
L_cm_ramienia = (m_c*L_c + m_p*L_p) / (m_r) 

#%% Pendulum center of mass position
xcm = x+L*sin(theta)
ycm = L*cos(theta)

#%% Pendulum forces
Dx_cm = xcm.diff(t, 1)
DDx_cm = xcm.diff(t, 2)
DDy_cm = ycm.diff(t, 2)

Rx_expr = m_r * DDx_cm - D*cos(alpha)
Ry_expr = m_r * DDy_cm + m_r*g + D*sin(alpha)

#%% Pendulum torques
Dtheta = theta.diff(t, 1)
DDtheta = theta.diff(t, 2)

lhs = J_cm*DDtheta
rhs = L*Ry*sin(theta) - L*Rx*cos(theta) - gamma*Dtheta + m_c*g*sin(theta)*(L_c-L) - m_p*g*sin(theta)*(L-L_p) + D*cos(smp.pi/4 - alpha)
bilans_momentow = smp.Eq(lhs=lhs, rhs=rhs)

#%% Rx & Ry substitution
f_trolley_eq = f_trolley_eq.subs([(Rx, Rx_expr),
                                  (Ry, Ry_expr)])
bilans_momentow = bilans_momentow.subs([(Rx, Rx_expr),
                                        (Ry, Ry_expr)])

#%% Pendulum torques after substitution
bilans_momentow.subs(m_r, m_c+m_p).expand().simplify()

#%% Moment of inertia of pendulum with respect to its center of mass
J_cm_calosciowe = smp.Rational(1, 3)*m_p*(2*L_p)**2 + m_c*L_c**2 - m_r*L**2
J_cm_calosciowe = J_cm_calosciowe.subs(L, L_cm_ramienia)


#%% 
Mt, Jt = smp.symbols('M_t J_t')

# Przepisać te macierze co u góry
Am = smp.Matrix([
    [Mt, m_r*L*cos(theta)],
    [m_r*L*cos(theta), Jt]
])
Bm = smp.Matrix([
    [DDx],
    [DDtheta]
])
Cm = smp.Matrix([
    [b*Dx - m_r*L*sin(theta)*Dtheta**2 ],
    [gamma*Dtheta - m_r*g*L*sin(theta)]
])
Dm = smp.Matrix([
    [u],
    [0]
])
Em = smp.Matrix([
    [D * cos(alpha)],
    [D * ( L*cos(alpha-theta) + sin(alpha + smp.pi/4) )]
])

DDx_DDtheta_rozwiazanie = Am.inv() * (Dm + Em - Cm)
x_ddot_result = DDx_DDtheta_rozwiazanie[0].simplify()
theta_ddot_result = DDx_DDtheta_rozwiazanie[1].simplify()

#%% State space representation
x1, x2, x3, x4 = smp.symbols('x_1 x_2 x_3 x_4', cls=smp.Function)
x1 = x1(t); x2 = x2(t); x3 = x3(t); x4 = x4(t) 

# state vector, x-owa notacja
state_vec = smp.Matrix([
    [x1],
    [x2],
    [x3],
    [x4]
])

# state vector
state_vec_noraml = smp.Matrix([
    [x],
    [theta],
    [Dx],
    [Dtheta]
])

# state equ
state_vec_diff = state_vec.diff(t)
state_vec_noraml_diff = state_vec_noraml.diff(t)

state_equations_normal = smp.Matrix([
    [Dx],
    [Dtheta],
    [x_ddot_result],
    [theta_ddot_result]
])

# pojawiały się babole jak "d/dt x2" zamiast x4
state_equations = state_equations_normal.subs([(Dx, x3),
                                               (Dtheta, x4),
                                               (x, x1),
                                               (theta, x2)])

#%% Symbolic model 
model_constants = {'L': smp.Eq(L, L_cm_ramienia),
                    'm_r': smp.Eq(m_r, m_p+m_c),
                    'Mt': smp.Eq(Mt, M+m_r),
                    'Jt': smp.Eq(Jt, J_cm + m_r*L**2),
                    'Jcm': smp.Eq(J_cm, smp.Rational(4, 3) * m_p * L_p**2 + m_c*L_c**2 + m_r*L**2)}

DAE_model = smp.Eq(
    smp.Add( smp.MatMul(Am, Bm), Cm),
    smp.Add(smp.Add(Dm,Em)
    )
)

ODE_model_var_natural = smp.Eq(state_vec_noraml_diff,
                               state_equations_normal)

ODE_model_var_x = smp.Eq(state_vec.diff(t, 1),
                         state_equations)

#%% Save
content_to_save = {
    'model_constants': model_constants,
    'DAE_model': DAE_model,
    'ODE_model_var_natural': ODE_model_var_natural,
    'ODE_model_var_x': ODE_model_var_x    
}

with open('symbolicznyModelWahadla.pkl', 'wb') as f:
    dill.dump(
        content_to_save,
        file=f,
        protocol=dill.DEFAULT_PROTOCOL,
        recurse=True)
    
'''
How to read the model
with open('symbolicznyModelWahadla.pkl', 'rb') as f:
    content = dill.load(f)
'''