'''
model with extra mass animation
AOF
'''
import numpy as np
from numpy import cos, sin
import scipy.integrate as integrate

class InvPendulum:
    def __init__(self, ic_state, params, origin, u_func):
        '''
        params = (M, mc, mp, Lp, Lc, g, b, gamma, D, alpha)        
        state  = (x, theta, Dx, Dtheta)
        origin = (x_origin, y_origin)
        
        Use params like this:
        M, mc, mp, Lp, Lc, g, b, gamma, D, alpha = self.params
        '''
        self.ic_state = np.array(ic_state, dtype=float)
        self.state = self.ic_state
        self.params = params
        self.time_elapsed = 0
        self.origin = origin
        self.u_func = u_func
        
        # Calculated parameters
        M, mc, mp, Lp, Lc, g, b, gamma, D, alpha = params
        
        # Calculated parameters
        self.mr = mc+mp
        self.Mt = self.mr + M
        self.L  = (Lc*mc + Lp*mp) / self.mr
        self.Jcm = (self.L**2)*self.mr + (Lc**2)*mc + 4/3 * (Lp**2)*mp
        self.Jt  = self.Jcm + self.mr*self.L**2

    def system_xy_pos(self):
        '''
        return: x_trolley, y_trolley, x_pend_end, y_pend_end
        state = (x, theta, Dx, Dtheta)
        '''
        
        lp = self.params[3]
        
        x_pen = self.origin[0] + 2*lp*sin(self.state[1]) + self.state[0]
        y_pen = self.origin[1] + 2*lp*cos(self.state[1])
        
        xs = [self.state[0], x_pen]        # (x_trolley, x_pendulum_end)
        ys = [self.origin[1], y_pen]       # (y_trolley, y_pendulum_end), y_trolley = y_origin
        
        # for ploting: plot([x, x_pen], [y_pen]) plots a line from trolley to pendulum end
        return (xs, ys)  
        
    def ddt_state(self, state, t, u_func):
        '''
        state = (x, theta, Dx, Dtheta)
        '''
        M, mc, mp, Lp, Lc, g, b, gamma, D, alpha = self.params
        mr, Mt, L, Jcm, Jt = (self.mr, self.Mt, self.L, self.Jcm, self.Jt)
        
        (x1, x2, x3, x4) = state        
        
        # u = u_func(t)
        u = u_func(t, state)    
        
        den = Jt*Mt - L**2 * mr**2 * cos(x2)

        # State eqs
        ddt_x1 = x3
        ddt_x2 = x4
        ddt_x3 = ( Jt*(D*cos(alpha) + L*mr*x4**2*sin(x2) - b*x3 + u) - \
                   L*mr*cos(x2) * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) \
                   / den
        
        ddt_x4 = ( -L*mr*cos(x2) * (D*cos(alpha) + L*mr*x4**2*sin(x2) - b*x3 + u) + \
                   Mt * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) \
                   / den



        return (ddt_x1, ddt_x2, ddt_x3, ddt_x4)
        
    def simulate_all(self, end_time, dt, Lp_scale=1):
        '''
        return: x_trolley, y_trolley, x_pen, y_pen, theta
        '''
        sim_time = np.arange(0, end_time+dt, dt)
        
        sol = integrate.odeint(func=self.ddt_state,
                               y0=self.ic_state,
                               t=sim_time,
                               args=(self.u_func, ),
                               h0=dt,
                               hmax=dt) # solver could optimize step size and input function u_func wont work 
        
        # (M, mr, L, Lp, Lc, g, b, gamma, Mt, Jt, D, alpha) = self.params
        Lp = self.params[3] * Lp_scale
        
        # odeint returns in form [[x1(0), x2(0), x3(0), x4(0)]
        #                         [x1(1), x2(1), x3(1), x4(1)] ... ]
        x_trolley, theta, x3_sol, x4_sol = sol.T
        
        y_trolley = np.ones_like(theta) * self.origin[1] # y is always zero, the same as origin y
        x_pen     = 2*Lp*sin(theta) + x_trolley
        y_pen     = 2*Lp*cos(theta)
        
        return np.array([x_trolley, y_trolley, x_pen, y_pen, theta])
