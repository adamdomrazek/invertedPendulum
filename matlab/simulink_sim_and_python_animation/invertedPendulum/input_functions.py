import numpy as np

'''
    Step function used in system simulation
'''
def step(t):
    return 1 if  t > 0 else 0

def u_2sec_step(t, state):
    ret = step(t-2)
    return ret

def u_basic_P_control_1(t, state, gain):
    ret = state[1] * gain    # u = 10 theta
    return ret