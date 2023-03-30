
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
import numpy as np
from invertedPendulum.invertedPendulum3_AOF import InvPendulum
# import matplotlib as mpl 
from invertedPendulum import input_functions as input_funcs
import functools 
from invertedPendulum import animate

M = 0.5 
mc = 0.1
mp = 0.1
Lc = 0.8
Lp = 0.1
g = 9.81
b = 0.001;
gamma = 0.001
D = 0
alpha = 0
params = (M, mc, mp, Lp, Lc, g, b, gamma, D, alpha)

origin = (0, 0) 

theta_ic = 3*np.pi/180
IC = [origin[0], theta_ic, 0, 0]

XLIM = (-0.4, 0.4)
YLIM = (-0.2, 0.2)

end_time = 6 
dt = 0.005
SAVE_DPI = 100

save = False
show_i = True

slow_down = False
slow_down_factor=2
filename = 'PENDULUM_ANIM_GIF.mp4'
extra_decimation = 1

blit = False

# u_func = lambda t: 0
u_func = functools.partial(input_funcs.u_basic_P_control_1, gain=0)
system = InvPendulum(ic_state=IC, params=params, origin=origin, u_func=u_func)
res = system.simulate_all(end_time=end_time, dt=dt)

###############################################
# aniamcja
###############################################
anim = animate.animo(xy_coords=res, figsize=(14, 8), end_time=end_time, dt=dt, XLIM=XLIM, YLIM=YLIM, pen_len_scale=0.3)
anim.show()