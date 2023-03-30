'''
    The best animation so far
    
    uses: 
    invertedPendulum / inveretdPendulum3_AOF.py
    
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axes.html
    
    TO DO:
    Add plot of trajectories of linearized system 
'''
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
import numpy as np
from invertedPendulum.invertedPendulum3_AOF import InvPendulum
# import matplotlib as mpl 
from invertedPendulum import input_functions as input_funcs
import functools 

M = 0.5 ; mc = 0.1 ; mp = 0.1
Lc = 0.8; Lp = 0.4
g = 9.81; b = 0.001; gamma = 0.001 #0.005
D = 0
alpha = 0
params = (M, mc, mp, Lp, Lc, g, b, gamma, D, alpha)

origin = (0, 0) 

theta_ic = 3*np.pi/180
IC = [origin[0], theta_ic, 0, 0]

XLIM = (-0.8, 0.8)
YLIM = (-0.4, 0.5)
Lp_scale = 0.4

end_time = 20      # [sek] 
dt = 0.005        # [sek]
SAVE_DPI = 100

save = False
show_i = True

slow_down = False
slow_down_factor=2
filename = 'PENDULUM_ANIM_GIF.mp4'

blit = False

# u_func = lambda t: 0
u_func = functools.partial(input_funcs.u_basic_P_control_1, gain=10)

system = InvPendulum(ic_state=IC, params=params, origin=origin, u_func=u_func)

# ==============================================================================================================
# AOT simulation
# System simulation output had to be downsampled so that simulated system state 
# will match simulation time 
# eg. simulation dt is 1 milli sec and animation step is 40 milli sec
# ==============================================================================================================
delay_between_frames_ms = 40
sample_decimation_factor = int(delay_between_frames_ms * 0.001 / dt) # eq. 50ms delay b/w frames dt=0.001 (1ms) -> 50*dt = 50ms
n_frames = int(end_time * 1000 / delay_between_frames_ms)

res = system.simulate_all(end_time=end_time, dt=dt, Lp_scale=Lp_scale)
print('ODE simulation: DONE')

# Decimation
x_trolley, y_trolley, x_pen, y_pen, theta = res[:, ::sample_decimation_factor]
theta_full_output = res[4]

# This simulation time vector is used only to set time text on the animation
t_vec = np.arange(0, end_time+dt, dt)

#
# Templete: http://apmonitor.com/do/index.php/Main/InvertedPendulum 
#
# plt.ioff()
plt.rcParams['animation.html'] = 'html5'
# plt.rcParams['lines.antialiased'] = True

fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8), facecolor='#caccca', dpi=50)

ax.set_aspect('equal')
ax.autoscale(0)
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_facecolor('#e3e6e3')
ax.get_yaxis().set_visible(False)   

ax.set_aspect('equal')
ax.autoscale(0)

crane_rail,        = ax.plot([-0.4, 0.4], [0, 0], 'k-', lw=8, zorder=0)
horizontal_line_0, = ax.plot([-XLIM[0], XLIM[0]], [0, 0], 'k--', lw=1, alpha=0.3, zorder=0)

# # vlines
# vertical_line_1, = ax.plot([-1,-1], [-1.5,1.5], 'k--', lw=1, alpha=0.3)
# vertical_line_2, = ax.plot([0,0], [-1.5,1.5], 'k--', lw=1, alpha=0.3)
# vertical_line_3, = ax.plot([1,1], [-1.5,1.5], 'k--', lw=1, alpha=0.3)

# Pendulum radius circle
pend_circle = Circle( (origin[0], origin[1]), 2*Lp*Lp_scale, fill=False, edgecolor='k', linestyle='--')
ax.add_artist(pend_circle)

trolley_block, = ax.plot([],[], linestyle='None', marker='s',
                 markersize=40, markeredgecolor='k',
                 color='orange', markeredgewidth=2, 
                 zorder=1)
mass2, = ax.plot([],[], linestyle='None', marker='o',
                 markersize=20, markeredgecolor='k',
                 color='orange',markeredgewidth=1,
                 zorder=1)
line, = ax.plot([],[], 'o-', color='black', lw=8,
                markersize=10, markeredgecolor='k',
                markerfacecolor='k',
                zorder=2)

time_template = 'time = {:.2f} [sek]'
time_text = ax.text(0.05, 0.8, '',transform=ax.transAxes)

def init():
    trolley_block.set_data([],[])
    mass2.set_data([],[])
    line.set_data([],[])
    pend_circle.center = (origin[0], origin[1])
    time_text.set_text('')
    
    return line, trolley_block, mass2, time_text

def animate(i):
    print('frame: {}/{}'.format(i, n_frames-1))
    
    trolley_block.set_data([x_trolley[i]], [y_trolley[i]])
    mass2.set_data([x_pen[i]], [y_pen[i]])
    line.set_data([x_trolley[i],x_pen[i]],[y_trolley[i],y_pen[i]])
    pend_circle.center = (x_trolley[i], y_trolley[i])
    time_text.set_text(time_template.format( t_vec[i * sample_decimation_factor] ))
    
    return line, trolley_block, mass2, time_text

ani_a = animation.FuncAnimation(fig=fig,
                                func=animate,
                                frames=n_frames,
                                interval=delay_between_frames_ms,
                                blit=blit,
                                init_func=init)

print('saving animation')
fps_save = 1/delay_between_frames_ms * 1000

if save:
    if slow_down:
        ani_a.save('./animations_2/'+filename,fps=fps_save/slow_down_factor, dpi=SAVE_DPI)
    else:
        ani_a.save('./animations_2/'+filename,fps=fps_save, dpi=SAVE_DPI)

if show_i:
    plt.show()

# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani_a.save("movie.mp4", writer=writer, dpi=10)

# plt.show() 