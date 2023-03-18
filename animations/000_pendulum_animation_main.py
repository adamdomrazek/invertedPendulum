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

#
# Pendlum = pendulum rod + extra mass at the end of the rod
#
# Pendulum parameters:
#
#   M       :   Cart mass 
#   mc      :   Extra mass at the end of pendulum rod
#   mp      :   Pendulum rod mass
#   mr      :   mp + mc
#   Lc      :   Distance from axis of rotation to extra mass
#   Lp      :   Half the length of pendulum rod (pendulum rod has uniform density so Lp is center of gravity of the rod)
#   L       :   Distance from axis of rotation to center of gravity of rod with extra mass
#   g       :   gravitational constant
#   b       :   Viscous friction constat
#   gamma   :   Rotational viscous friction constant
#   Jcm     :   Moment of inertia of pendulum about its center of gravity (L from axis of rotation) 
#   Mt      :   Total mass of the system, Mt = M + mr
#   Jt      :   Total moment of inertia of the pendulum (about its axis of rotation)
#   D       :   Disturbance force magnitude at the end of the rod 
#   alpha   :   angle of the disturbance force vector from horizontal
#   
# These are calculated inside invPendulum class
# mr = mc+mp
# Mt = M+mr
# L = (Lc*mc + Lp*mp) / mr
# Jcm = (L**2)*mr + (Lc**2)*mc + 4/3 * (Lp**2)*mp
# Jt  = Jcm + mr*L**2
#
M = 2 ; mc = 0.1 ; mp = 0.1
Lc = 0.8; Lp = 0.4;
g = 9.81; b = 0.001; gamma = 0.001 #0.005
D = 0
alpha = 0
params = (M, mc, mp, Lp, Lc, g, b, gamma, D, alpha)

# 
# Trolley starting position
# 
origin = (0, 0) 

# 
# Initial conditions = [x, theta, x_dot, theta_dot]
# 
theta_ic = 0.1*np.pi/180
IC = [origin[0], theta_ic, 0, 0]

#
# Cart animation axes limits 
#
XLIM = (-6, 6)
YLIM = (-1, 1)

#
# Simulation time and step size for integration
#
end_time = 12      # [sek] 
dt = 0.001        # [sek]

#
# Input signal function
#
from invertedPendulum.step import * # there's only one function in there

u_func = lambda t: 3 * step(t-4)

# u_func = lambda t: 0

# u_func = lambda t: 10*step(t-0.3) - 10*step(t-0.4) + \
#                    ( - 15*step(t-2) + 15*step(t-2.3)  ) * ( 0.5*np.sin(2*np.pi*3*t) + 1)

# # jiggle the cart
# ampl = 15 
# freq = 2
# u_func = lambda t: ampl*np.sin(2*np.pi*freq*t) * (step(t-0.5) - step(t-3.5))  

#
# System instance
# 
system = InvPendulum(ic_state=IC, params=params, origin=origin, u_func=u_func)

#
# AOT simulation
# System simulation output had to be downsampled so that simulated system state 
# will match simulation time 
# eg. simulation dt is 1 milli sec and animation step is 40 milli sec
#
delay_between_frames_ms = 40
sample_decimation_factor = int(delay_between_frames_ms * 0.001 / dt) # eq. 50ms delay b/w frames dt=0.001 (1ms) -> 50*dt = 50ms
n_frames = int(end_time * 1000 / delay_between_frames_ms)

res = system.simulate_all(end_time=end_time, dt=dt)
print('ODE simulation: DONE')
x_trolley, y_trolley, x_pen, y_pen, theta = res[:, ::sample_decimation_factor]
theta_full_output = res[4]

# This simulation time vector is used only to set time text on the animation
t_vec = np.arange(0, end_time+dt, dt)

#
# Templete: http://apmonitor.com/do/index.php/Main/InvertedPendulum 
#
plt.rcParams['animation.html'] = 'html5'
# plt.rcParams['text.u  setex'] = True    # massive slow down, use only while saving simulation
fig = plt.figure(figsize=(14,8), facecolor='#caccca')

# 2 by 2 axes grid 
gs = fig.add_gridspec(2, 2)

#
# Main top axes
#
ax = fig.add_subplot(gs[0, :],
                     aspect='equal',
                     autoscale_on=False,
                     xlim=XLIM,
                     ylim=YLIM)
ax.set_facecolor('#e3e6e3')
ax.set_xlabel('Cart position position x(t) [m]')
ax.get_yaxis().set_visible(False)   

crane_rail,        = ax.plot([-XLIM[0], XLIM[0]], [-0.2,-0.2], 'k-', lw=8)
horizontal_line_0, = ax.plot([-XLIM[0], XLIM[0]], [0, 0], 'k--', lw=1, alpha=0.3)

# vlines
vertical_line_1, = ax.plot([-1,-1], [-1.5,1.5], 'k--', lw=1, alpha=0.3)
vertical_line_2, = ax.plot([0,0], [-1.5,1.5], 'k--', lw=1, alpha=0.3)
vertical_line_3, = ax.plot([1,1], [-1.5,1.5], 'k--', lw=1, alpha=0.3)

# Pendulum radius circle
pend_circle = Circle( (origin[0], origin[1]), 2*Lp, fill=False, edgecolor='k', linestyle='--')
ax.add_artist(pend_circle)

mass1, = ax.plot([],[], linestyle='None', marker='s',
                 markersize=40, markeredgecolor='k',
                 color='orange', markeredgewidth=2)
mass2, = ax.plot([],[], linestyle='None', marker='o',
                 markersize=20, markeredgecolor='k',
                 color='orange',markeredgewidth=2)
line, = ax.plot([],[], 'o-', color='black', lw=8,
                markersize=10, markeredgecolor='k',
                markerfacecolor='k')

time_template = 'time = {:.2f} [sek]'
time_text = ax.text(0.05, 0.8, '',transform=ax.transAxes)
vline_1_text = ax.text(-1.06,-0.3,'',ha='right') # w oryginalnym tutorialu było to 'objective' i 'start
vline_2_text = ax.text(-0.06,-0.3,'',ha='right') # tutorial był o optymalizacji dynamicznej
vline_3_text = ax.text(0.94,-0.3,'',ha='right')

#
# Bottom left axes
#
ax_bot = fig.add_subplot(gs[1, 0],
                         autoscale_on=False,
                         xlim=(0, t_vec[-1]),
                         ylim=(-20, 20))

ax_bot.set_facecolor('#e3e6e3')
ax_bot.set_ylabel('Input signal amplitude')
ax_bot.set_xlabel('Time t [sec]')
ax_bot.grid()

u_line_data = list( map(u_func, t_vec) )
uline, = ax_bot.plot(t_vec, u_line_data,
                     color='black', lw=1, linestyle='--')

moving_dot, = ax_bot.plot([],[], linestyle='None', marker='.',
                          markersize=30, markeredgecolor='k', 
                          color='red', markeredgewidth=4)

#
# Bottom axes 2
#
ax_bot_2 = fig.add_subplot(gs[1, 1],
                           autoscale_on=False,
                           xlim=(0, t_vec[-1]),
                           ylim=(0, 6))

ax_bot_2.set_facecolor('#e3e6e3')
ax_bot_2.set_ylabel('Theta')
ax_bot_2.set_xlabel('Time t [sec]')
ax_bot_2.grid()

theta_line, = ax_bot_2.plot(t_vec, theta_full_output,
                          color='black', lw=1, linestyle='--')

moving_dot_2, = ax_bot_2.plot([],[], linestyle='None', marker='.',
                          markersize=30, markeredgecolor='k', 
                          color='red', markeredgewidth=4)

def init():
    #
    # Top axes
    #
    mass1.set_data([],[])
    mass2.set_data([],[])
    line.set_data([],[])
    pend_circle.center = (origin[0], origin[1])
    time_text.set_text('')
    
    #
    # Bottom axes
    #
    moving_dot.set_data([], [])
    moving_dot_2.set_data([], [])
    
    return line, mass1, mass2, time_text

def animate(i):
    print('frame: {}/{}'.format(i, n_frames-1))
    #
    # Main top axes
    #
    mass1.set_data([x_trolley[i]], [y_trolley[i]-0.1])
    mass2.set_data([x_pen[i]], [y_pen[i]])
    line.set_data([x_trolley[i],x_pen[i]],[y_trolley[i],y_pen[i]])
    pend_circle.center = (x_trolley[i], y_trolley[i])
    time_text.set_text(time_template.format( t_vec[i * sample_decimation_factor] ))
    
    #
    # Bottom axes
    #
    moving_dot.set_data(t_vec[i * sample_decimation_factor], u_line_data[i * sample_decimation_factor])
    moving_dot_2.set_data(t_vec[i * sample_decimation_factor], theta_full_output[i * sample_decimation_factor])
    
    return line, mass1, mass2, time_text

ani_a = animation.FuncAnimation(fig=fig,
                                func=animate,
                                frames=n_frames,
                                interval=delay_between_frames_ms,
                                blit=False,
                                init_func=init)

print('saving animation')
slow_down = False
fps_save = 1/delay_between_frames_ms * 1000
if slow_down:
    ani_a.save('./animations/PENDULUM_ANIM_GIF.gif',fps=fps_save/2)
else:
    ani_a.save('./animations/PENDULUM_ANIM_GIF.gif',fps=fps_save)
    
# plt.show() 