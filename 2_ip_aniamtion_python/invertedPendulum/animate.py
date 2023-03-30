import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from functools import partial
import numpy as np

class animo:
    def __init__(self, xy_coords, figsize=(14, 8), end_time=10, dt=0.001, XLIM=(-1, 1), YLIM=(-1, 1), pen_len_scale=1):
        delay_between_frames_ms = 40
        self.sample_decimation_factor = int(delay_between_frames_ms * 0.001 / dt)
        self.n_frames = int(end_time * 1000 / delay_between_frames_ms)
        self.xy_coords = xy_coords
        
        self.fig = plt.figure(figsize=figsize, facecolor='#caccca', dpi=50)

        self.xlim = XLIM
        self.ylim = YLIM
        

        self.ax, self.trolley_block, self.mass2, self.line, self.time_text, self.time_template = self.setupFigure()
    
        
        self.anim = animation.FuncAnimation(fig=self.fig,
                                       func=self.animate,
                                       frames=self.n_frames,
                                       interval=delay_between_frames_ms,
                                       blit=True,
                                       init_func=self.init)
        
        self.x_trolley, self.y_trolley, self.x_pen, self.y_pen, self.theta = xy_coords[:, ::self.sample_decimation_factor]
    
        self.t_vec = np.arange(0, end_time+dt, dt)

    
    def show(self):
        plt.show()
    
    def setupFigure(self):
        ax = self.fig.add_subplot(111,
                             aspect='equal',
                             autoscale_on=False,
                             xlim=self.xlim,
                             ylim=self.ylim)
        ax.set_facecolor('#e3e6e3')
        # ax.get_yaxis().set_visible(False)  

        crane_rail,        = ax.plot((self.xlim[0], self.xlim[1]), [-0.2,-0.2], 'k-', lw=8, zorder=0)
        # horizontal_line_0, = ax.plot((-6, 6), [0, 0], 'k--', lw=1, alpha=0.3, zorder=0)

        # pend_circle = Circle( (0, 0), 2*Lp, fill=False, edgecolor='k', linestyle='--')
        # ax.add_artist(pend_circle)

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
        
        return ax, trolley_block, mass2, line, time_text, time_template

    def init(self):
        self.trolley_block.set_data([],[])
        self.mass2.set_data([],[])
        self.line.set_data([],[])
        # self.pend_circle.center = (self.origin[0], self.origin[1])
        self.time_text.set_text('')
        
        return self.line, self.trolley_block, self.mass2, self.time_text
    
    def animate(self, i):
        print('frame: {}/{}'.format(i, self.n_frames-1))
        
        self.trolley_block.set_data([self.x_trolley[i]], [self.y_trolley[i]])
        self.mass2.set_data([self.x_pen[i]], [self.y_pen[i]])
        self.line.set_data([self.x_trolley[i], self.x_pen[i]], [self.y_trolley[i], self.y_pen[i]])
        # self.pend_circle.center = (self.x_trolley[i], self.y_trolley[i])
        self.time_text.set_text(self.time_template.format( self.t_vec[i * self.sample_decimation_factor] ))
        
        return self.line, self.trolley_block, self.mass2, self.time_text