"""
    Code used for speedtest in ZZ_speed_test.py
"""

# ================================================================================
# SETUP-START
####################
Ts = 0.01   # 10ms
end_time = 20
####################

import numpy as np
import scipy.integrate as integrate
from invertedPendulum.invertedPendulum2 import InvPendulum

# params
M = 0.1
mc = 0.1
mp = 0.1
Lc = 0.3
Lp = 0.15
mr = mc+mp
g = 9.81
b = 0.001
gamma = 0.005
Mt = M+mr
L = (Lc*mc + Lp*mp) / mr
Jcm = (L**2)*mr + (Lc**2)*mc + 4/3 * (Lp**2)*mp
Jt  = Jcm + mr*L**2
# zaklucenia
D = 0
alpha = 0

params = (M, mr, L, g, b, gamma, Mt, Jt, D, alpha)

# Trolley center
origin = (0, 0) 

# IC
x_ic        = origin[0]
theta_ic    = 170 * np.pi / 180
D_x_ic      = 0
D_theta_ic  = 0 * np.pi / 180
IC = [x_ic, theta_ic, D_x_ic, D_theta_ic]

# Wymuszenie
step = lambda t: 1 if t > 1 else 0
u_func = lambda t: 5*step(t-3) - 5*step(t-3.1) + (-8*step(t-3.6) + 8*(step(t-3.7))) - step(t-5)*3*np.sin(2*np.pi*t)

# System
system = InvPendulum(ic_state=IC, params=params, origin=origin, u_func=u_func)
sim_time = np.arange(0, end_time, Ts)

# SETUP-END
# ================================================================================

# ================================================================================
# CODE-START

sol = integrate.odeint(func=system.ddt_state,
                       y0=IC,
                       t=sim_time,
                       args=(u_func, ))

# CODE-END
# ================================================================================