% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script is an interface between simulink inverted pendulum simulation
% and animation written in python
%
% model settings / algebraic loops: none
% min step size : 0.01
%
% Pendlum = pendulum rod + extra mass at the end of the rod
%
% Pendulum parameters:
%
%   M       :   Cart mass
%   mc      :   Extra mass at the end of pendulum rod
%   mp      :   Pendulum rod mass
%   mr      :   mp + mc
%   Lc      :   Distance from axis of rotation to extra mass
%   Lp      :   Half the length of pendulum rod (pendulum rod has uniform density so Lp is center of gravity of the rod)
%   L       :   Distance from axis of rotation to center of gravity of rod with extra mass
%   g       :   gravitational constant
%   b       :   Viscous friction constat
%   gamma   :   Rotational viscous friction constant
%   Jcm     :   Moment of inertia of pendulum about its center of gravity (L from axis of rotation)
%   Mt      :   Total mass of the system, Mt = M + mr
%   Jt      :   Total moment of inertia of the pendulum (about its axis of rotation)
%   D       :   Disturbance force magnitude at the end of the rod
%   alpha   :   angle of the disturbance force vector from horizontal
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
% dbstop if error
clc
clear

to_plot_or_not_to_plot = true;

%% Change simulink cache folders temporarly
cfg = Simulink.fileGenControl('getConfig');
cfg.CacheFolder = fullfile(eval(['pwd']),'generatedCacheSimulink');
cfg.CodeGenFolder = fullfile(eval(['pwd']),'generatedCodeSimulink');
Simulink.fileGenControl('setConfig', 'config', cfg,'createDir',true);
% Simulink.FileGenConfig -> to check if default cache dirs settings are set
% after this session

%% pseudoregulator
K_x        = 0.01;
K_theta    = 30;
K_Dx       = 0.01;
K_Dtheta   = 3;

%% Input for python script
% Initial conditions
x_ic        = 0;
theta_ic    = 3 * pi/180;
D_x_ic      = 0;
D_theta_ic  = 0;

% left-hand side names are the same as in python script
IC                  = [x_ic, theta_ic, D_x_ic, D_theta_ic]';                  
XLIM                = '(-0.5, 0.5)';
YLIM                = '(-0.5, 0.5)';
end_time            = 30;
dt                  = 0.001;
SAVE_DPI            = 140;
PLAY_DPI            = 100;
Lp_scale            = 0.17; % Scales pendulum rod in length, only for animation
ROD_SCALE           = 0.14;
MASS2_SCALE         = 2;
TROLLEY_SCALE       = 0.5;
save_anim           = 'False';
show_i              = 'True'; % if true, show animation with plt.show
slow_down           = 'False';
slow_down_factor    = 2;
filename            = 'PENDULUM_ANIM_GIF.mp4';
extra_decimation    = 1; % constant used for state vector decimation, used for tests
blit                = 'False'; % don't use blit in matlab, matplotlib opens 4 windows
origin              = '(0, 0)';  
figsize             = '(15, 7)';

%% Python setup
use_default_python = true;
run('a1_python_setup.m');

%% Input signals
time = 0:dt:end_time;
u1 = ustep(time, 3);
u2 = ustep(time, 4);
u3 = ustep(time, 4) * 0.5;
u4 = ustep(time, 10) * 0.5;
u5 = ustep(time, 10) * -2;
u6 = ustep(time, 15) * -2;
ref_sig = u1 - u2 + u3 - u4 + u5 - u6;
ref_sig = ref_sig * 0.5;

% Structure with time; input to simulink model
U_sim.time = time;
U_sim.signals.values(:, 1) = ref_sig;
U_sim.signals.dimensions = 1;

%%
% Input signal
% plot(U_sim.time, U_sim.signals.values)

%%
M = 0.5 ; mc = 0.1 ; mp = 0.1;
Lc = 0.8; Lp = 0.4;
g = 9.81; b = 0.001; gamma = 0.001;
D = 0;
alpha = 0;

% Calculated parameters
mr = mc + mp;
Mt = mr + M;
L  = (Lc*mc + Lp*mp) / mr;
Jcm = (L^2)*mr + (Lc^2)*mc + 4/3 * (Lp^2)*mp;
Jt  = Jcm + mr*L^2;

% Params for pendulum function
params = [M, mc, mp, Lp, Lc, g, b, gamma, D, alpha, mr, Mt, L, Jcm, Jt];
% params = struct('values', {params});

%% Sim
simout = sim('a2_sim_model.slx');

%% Results
states  = simout.state_out.signals.values';
tout    = simout.tout';
u       = simout.u.Data;

results.states = states;
results.tout   = tout;
results.u      = u;

save('simout.mat', 'results');

%% gfx
if to_plot_or_not_to_plot
    clf
    close
    run('a4_plot.m')
end

%%
    x       = states(1, :);
theta   = states(2, :);
Dx      = states(3, :);
Dtheta  = states(4, :);

pyrunfile('a5_py_animation.py', x=x, theta=theta, Dx=Dx, Dtheta=Dtheta, ...
                                  IC=IC, XLIM=XLIM, YLIM=YLIM, ...
                                  end_time=end_time, dt=dt, SAVE_DPI=SAVE_DPI, ...
                                  save_anim=save_anim, show_i=show_i, slow_down=slow_down, ...
                                  slow_down_factor=slow_down_factor, ...
                                  filename=filename, extra_decimation=extra_decimation, ...
                                  blit=blit, Lp=Lp, origin=origin, figsize=figsize, PLAY_DPI=PLAY_DPI, Lp_scale=Lp_scale, ...
                                  TROLLEY_SCALE=TROLLEY_SCALE, MASS2_SCALE=MASS2_SCALE, ROD_SCALE=ROD_SCALE);
















