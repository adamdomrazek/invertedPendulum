'''
    Variables from matlab:
                                matlab type:
        IC                      
        XLIM
        YLIM
        end_time
        dt
        SAVE_DPI
        save_anim
        show_i
        slow_down
        slow_down_factor
        filename
        extra_decimation
        blit
        Lp
        origin
        figsize
        
        states:
            x
            theta 
            Dx
            Dtheta
            
        i jakieś jeszcze dodatkowe
'''

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
import numpy as np
from numpy import cos, sin
# import functools 
# import sys

############################################
#   Some args from matlab are passed as strings
############################################
IC          = np.array(IC)
XLIM        = eval(XLIM)
YLIM        = eval(YLIM)
save_anim   = eval(save_anim)
show_i      = eval(show_i)
slow_down   = eval(slow_down)
blit        = eval(blit)
origin      = eval(origin)
figsize     = eval(figsize)


############################################
#   ODE solution setup
############################################
delay_between_frames_ms     = 40
sample_decimation_factor    = int(extra_decimation*delay_between_frames_ms * 0.001 / dt) # eq. 50ms delay b/w frames dt=0.001 (1ms) -> 50*dt = 50ms
n_frames                    = int(end_time * 1000 / delay_between_frames_ms)

states = np.array([x, theta, Dx, Dtheta])
states = states[:, ::sample_decimation_factor]  # decimation

############################################
#   trolley and pendulum tip x-y positions
############################################
Lp = Lp*Lp_scale
x_trolley, theta, _, _ = states 
y_trolley              = np.ones_like(theta) * origin[1]
x_pen                  = 2*Lp*sin(theta) + x_trolley
y_pen                  = 2*Lp*cos(theta)

t_vec = np.arange(0, end_time+dt, dt)

############################################
#   Figure setup
############################################
plt.rcParams['animation.html'] = 'html5'
# plt.rcParams['lines.antialiased'] = True

# gs = fig.add_gridspec(2, 2)
fig = plt.figure(figsize=figsize, facecolor='#caccca', dpi=PLAY_DPI)

# Left axis
ax = fig.add_subplot(111,
                     aspect='equal',
                     autoscale_on=False,
                     xlim=XLIM,
                     ylim=YLIM)
ax.set_facecolor('#e3e6e3')
ax.get_yaxis().set_visible(False)   

crane_rail,        = ax.plot([-XLIM[0], XLIM[0]], [-0.2,-0.2], 'k-', lw=8, zorder=0)
horizontal_line_0, = ax.plot([-XLIM[0], XLIM[0]], [1, 1], 'k--', lw=1, alpha=0.3, zorder=0)

# # vlines
# vertical_line_1, = ax.plot([-1,-1], [-1.5,1.5], 'k--', lw=1, alpha=0.3)
# vertical_line_2, = ax.plot([0,0], [-1.5,1.5], 'k--', lw=1, alpha=0.3)
# vertical_line_3, = ax.plot([1,1], [-1.5,1.5], 'k--', lw=1, alpha=0.3)

# Pendulum radius circle
pend_circle = Circle( (origin[0], origin[1]), 2*Lp, fill=False, edgecolor='k', linestyle='--')
ax.add_artist(pend_circle)

trolley_block, = ax.plot([],[], linestyle='None', marker='s',
                         markersize=int(40*TROLLEY_SCALE), markeredgecolor='k',
                         color='orange', markeredgewidth=2, 
                         zorder=1)
mass2, = ax.plot([],[], linestyle='None', marker='o',
                 markersize=int(10*MASS2_SCALE), markeredgecolor='k',
                 color='orange',markeredgewidth=1,
                 zorder=3)
line, = ax.plot([],[], 'o-', color='black', lw=8,
                markersize=int(10*ROD_SCALE), markeredgecolor='k',
                markerfacecolor='k',
                zorder=2)

time_template   = 'time = {:.2f} [sek]'
time_text       = ax.text(0.05, 0.8, '',transform=ax.transAxes)
    
############################################
#   Animation functions
############################################
def init():
    trolley_block.set_data([],[])
    mass2.set_data([],[])
    line.set_data([],[])
    pend_circle.center = (origin[0], origin[1])
    time_text.set_text('')
    
    return line, trolley_block, mass2, time_text

def animate(i):
    print('frame: {}/{}'.format(i, n_frames-1))
    
    trolley_block.set_data([x_trolley[i]], [y_trolley[i] - 0.2])
    mass2.set_data([x_pen[i]], [y_pen[i]-0.2])
    line.set_data([x_trolley[i],x_pen[i]],[y_trolley[i]- 0.2,y_pen[i]- 0.2])
    pend_circle.center = (x_trolley[i], y_trolley[i]-0.2)
    time_text.set_text(time_template.format( t_vec[i * sample_decimation_factor] ))
    
    return line, trolley_block, mass2, time_text

############################################
#   Animation object
############################################
ani_a = animation.FuncAnimation(fig=fig,
                                func=animate,
                                frames=n_frames,
                                interval=delay_between_frames_ms,
                                blit=blit,
                                init_func=init)


############################################
#   Saving animation
############################################
print('saving animation')
fps_save = 1/delay_between_frames_ms * 1000
import os
print('savign as: '+ os.getcwd() + '\\' + filename)
if save_anim:
    if slow_down:
        ani_a.save('./'+filename,fps=fps_save/slow_down_factor, dpi=SAVE_DPI)
    else:
        ani_a.save('./'+filename,fps=fps_save, dpi=SAVE_DPI)

if show_i:
    plt.show()
    

    