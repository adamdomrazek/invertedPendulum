import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from numpy import sin, cos
import numpy as np
import matplotlib as mpl
import itertools as it
from scipy import integrate
from math import ceil


def main():
    # Stałe i parametry symluacji 
    M = 0.1; mc = 0.1; mp = 0.1
    Lc = 0.8; Lp = 0.4
    g = 9.81; b = 0.001; gamma = 0.5 #0.005
    D = 0
    alpha = 0
    mr = mc+mp
    Mt = mr + M
    L  = (Lc*mc + Lp*mp) / mr
    Jcm = (L**2)*mr + (Lc**2)*mc + 4/3 * (Lp**2)*mp
    Jt  = Jcm + mr*L**2
    params = (M, mc, mp, Lp, Lc, g, b, gamma, D, alpha, mr, Mt, L, Jcm, Jt)
    u_func = lambda t: 0
    
    # v field
    x, theta, Dx, Dtheta = generate_sample_trajectories(u_func, params)
    theta_cmat, D_theta_cmat, Dx_cmat, dir_x2, dir_x4, dir_x3 = gen_v_field(params)
    multi_vec_field_slice_viewer(theta_cmat, Dx_cmat, D_theta_cmat, dir_x2, dir_x3, dir_x4)
    
    # # brain
    # struct_arr = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
    # images_T = struct_arr.T
    # multi_slice_viewer(images_T)
    
    plt.show()

# ################################################################
# Do pola wektorowego
# ################################################################

# Konwertuje wartości w tablicy na odpowiadające indeksy
def get_idx_from_vals(maxv, minv, step, value):
    '''
        Konwertuje wartość zmiennej na jej indeks w odpowiadającej jej tablicy
        np. x_arr = [1.2, 1.4, 1.6, 1.8]
            get_idx_from_vals(1.4) >>> returns: idx = 1 
    '''
    arr_len = (maxv - minv) / step + 1
    slope = (arr_len - 1) / (maxv - minv)
    idx = lambda elem: ceil(slope*(elem - minv))
    
    idx = ceil(slope*(value - minv))
    
    return idx

    # eg. use #
    # if 0:
    #     # od 0 do 1 co 0.1
    #     temp = np.arange(-3, 1.1, 0.1)
    #     print(list( enumerate(temp) )[30:37])
    #     print('number .5 index is: ', get_idx_from_vals(1, -3, 0.1, 0.5))

# Funkcja do scipy.odeint -> samplowe trajektorie
def ddt_state(state, t, u_func, params):
    '''
        Function used for scipy.integrate.odeint to generate sample state trajectory (ode solution)
        state = ( x  theta  Dx  Dtheta )
                ( x1  x2    x3    x4   )
    '''
    x1, x2, x3, x4 = state
    M, mc, mp, Lp, Lc, g, b, gamma, D, alpha, mr, Mt, L, Jcm, Jt = params

    u=u_func(t)
    den = Jt*Mt - L**2 * mr**2 * cos(x2)

    ddt_x1 = x3
    ddt_x2 = x4
    ddt_x3 = ( Jt*(D*cos(alpha) + L*mr*x4**2*sin(x2) - b*x3 + u) - \
                L*mr*cos(x2) * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) \
                / den
    
    ddt_x4 = ( -L*mr*cos(x2) * (D*cos(alpha) + L*mr*x4**2*sin(x2) - b*x3 + u) + \
                Mt * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) \
                / den
    
    return (ddt_x1, ddt_x2, ddt_x3, ddt_x4)

# Zwraca funkcje do obliczania macierzy z komponentami pola wektorowego
def ddt_states_for_v_field(params):
    ''' 
        x1    x2      x3      x4
        x    theta    Dx    Dtheta
        
        Returns 4 state functions ddt_x1 ddt_x2 ddt_x3 ddt_x4 as a function of state (u=0).
        These 4 functions can be used to generate slices of 4D vector field (of 4D state space)
        
        ddt_x1 is not a function of any state except x3 so:
            x1(x3) - constant slope
            (has constant slope and x1 intercept depends on initial conditions) 
        
        ddt_x2 - the same as in ddt_x1 case:
            x2(x4) - constant slope
    '''
    M, mc, mp, Lp, Lc, g, b, gamma, D, alpha, mr, Mt, L, Jcm, Jt = params

    u=0
    
    def ddt_x1(x1, x2, x3, x4):
        return x3
    def ddt_x2(x1, x2, x3, x4):
        return x4
    def ddt_x3(x1, x2, x3, x4):
        u=0
        ret = ( Jt*(D*cos(alpha) + L*mr*np.power(x4, 2)*sin(x2) - b*x3 + u) - \
                L*mr*cos(x2) * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) \
                / (Jt*Mt - L**2 * mr**2 * cos(x2))
        return ret
    
    def ddt_x4(x1, x2, x3, x4):
        u=0
        ret = ( -L*mr*cos(x2) * (D*cos(alpha) + L*mr*np.power(x4, 2)*sin(x2) - b*x3 + u) + \
                Mt * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) \
                / (Jt*Mt - L**2 * mr**2 * cos(x2))
        return ret

    return (ddt_x1, ddt_x2, ddt_x3, ddt_x4) 

# Samplowe trajektorie
def generate_sample_trajectories(u_func, params):
    dt = 0.001
    sim_time = np.arange(0, 10+dt, dt)

    # Warunki początkowe
    ic_states = np.array([
        [0, 10*np.pi/180, 0, 0],
        [0, -10*np.pi/180, 0, 0],
        [0, 100*np.pi/180, 0, 140*np.pi/180],
        [0, 200*np.pi/180, 0, -150*np.pi/180]
    ])

    # czasami gdy nie ma ustawionego hmax, solver
    # optymalizuje dt i wtedy u_func nie działa poprawnie
    # u_func gdy ma 'step(t)' jest nieciągła - to chyba dla tego
    sols = []
    for ic in ic_states:
        state_trajectory = integrate.odeint(
            func=ddt_state,
            y0=ic,
            t=sim_time,
            args=(u_func, params),
            h0=dt,
            hmax=dt) 
        sols.append(state_trajectory)    

    sols = np.transpose(sols)
    x, theta, Dx, Dtheta = sols

    # rad to deg
    theta = theta * 180/np.pi
    Dtheta = Dtheta * 180/np.pi
    
    return (x, theta, Dx, Dtheta)

# Generowanie Stacku pól wektorowych
def gen_v_field(params):
    xlims = np.array([-450, 450])
    ylims = np.array([-300, 300]) 

    grid_tick_step = 50

    deg_theta_step = 15
    deg_Dtheta_step = 15
    
    # convert degrees settings to radians
    x1_v_step, x2_v_step, x3_v_step, x4_v_step = (0.2,
                                                  deg_theta_step*np.pi/180,
                                                  0.2,
                                                  deg_Dtheta_step*np.pi/180)

    xlims_rad = xlims * np.pi / 180
    ylims_rad = ylims * np.pi / 180

    x1_const = 0                                          # x
    x2_coord = np.arange(xlims_rad[0], xlims_rad[1]+x2_v_step, x2_v_step)    # theta
    x3_coord = np.arange(0, 1000, 10)                                           # Dx       Prędkość od 0 do 10 co 1
    x4_coord = np.arange(ylims_rad[0], ylims_rad[1]+x4_v_step, x4_v_step)    # Dtheta
    
    # dla obu: dim = 2x2 (dwie osie, tj. ndim=2)
    # To stworzy trójwymiarowe macierze z koordynatami
    # w przestrzeni (theta, Dx, Dtheta)
    # wszystkie 3 macierze mają ndim=3 (mają trzy osie, indeksowanie naturalne)
    theta_cmat, Dx_cmat, D_theta_cmat = np.meshgrid(x2_coord, x3_coord, x4_coord, indexing='ij')

    d_x1_fun, d_x2_fun, d_x3_fun, d_x4_fun = ddt_states_for_v_field(params=params)

    # dir_x1 = d_x1_fun(x1_const, theta_cmat, Dx_cmat, D_theta_cmat)
    dir_x2 = d_x2_fun(x1_const, theta_cmat, Dx_cmat, D_theta_cmat)
    dir_x3 = d_x3_fun(x1_const, theta_cmat, Dx_cmat, D_theta_cmat)
    dir_x4 = d_x4_fun(x1_const, theta_cmat, Dx_cmat, D_theta_cmat)
    
    # pole wektorowe jest zbiorem wektorów o elementach:
    # < dir_x2, dir_x3, dir_x4 > tylko że wszystkie te macierze mają 3 osie
    
    # convert results in radians to results in degrees
    theta_cmat = theta_cmat * 180/np.pi
    D_theta_cmat = D_theta_cmat * 180/np.pi
    dir_x2 = dir_x2 * 180/np.pi
    dir_x4 = dir_x4 * 180/np.pi
    
    vec_len = np.sqrt( np.power(dir_x2, 2) + np.power(dir_x4, 2) + np.power(dir_x3, 2) )
    
    scale = 5 # to jest ważne bo skaluje długość wektora do wizualizacji
    dir_x2 = dir_x2/vec_len * scale
    dir_x4 = dir_x4/vec_len * scale
    dir_x3 = dir_x3/vec_len * scale
    
    return (theta_cmat, D_theta_cmat, Dx_cmat, dir_x2, dir_x4, dir_x3)

# ################################################################ 
# Obsługa klawiszy
# ################################################################

# usuwa defaultową funkcjonalność klawisza k
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index, :, :], interpolation='hanning')
    fig.canvas.mpl_connect('key_press_event', process_key_brain)
    
def multi_vec_field_slice_viewer(x2, x3, x4, d2, d3, d4):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.index = 0
    ax.data_len = x3.shape[1]
    print(x3.shape[1])
    # meshgrid(x2, x3, x4)
    ax.quiver(x2[ :, ax.index, : ], x4[ :, ax.index, : ], d2[ :, ax.index, : ], d4[ :, ax.index, : ],
              angles='xy', scale_units='xy',
              scale=0.32,
              width=0.001,       # szerokość pręta
              headwidth=5.5,
              headaxislength=9,  # wcięcie
              headlength=9,
              color='blue',
              zorder=10,
              alpha=0.3)
    
    # ax.quiver(xx, yy, uu, vv,
    #           angles='xy', scale_units='xy',
    #           scale=0.32,
    #           width=0.001,       # szerokość pręta
    #           headwidth=5.5,
    #           headaxislength=9,  # wcięcie
    #           headlength=9,
    #           color='blue',
    #           zorder=10,
    #           alpha=0.3)
    ax.set_title(f'index: {ax.index}')
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    ax.set_title(f'idx: {ax.index}')
    fig.canvas.draw()

def process_key_brain(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice_brain(ax)
    elif event.key == 'k':
        next_slice_brain(ax)
    ax.set_title(f'idx: {ax.index}')
    fig.canvas.draw()
        

def previous_slice(ax):
    """Go to the previous slice."""
    data_len = ax.data_len
    ax.index = (ax.index - 1) % data_len  # wrap around using %

def next_slice(ax):
    """Go to the next slice."""
    data_len = ax.data_len
    ax.index = (ax.index + 1) % data_len
   
def previous_slice_brain(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index]) 

def next_slice_brain(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    
if __name__ == '__main__':
    
    main()