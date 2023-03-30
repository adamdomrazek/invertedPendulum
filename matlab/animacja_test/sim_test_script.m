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
K_x        = 0;
K_theta    = 0;
K_Dx       = 0;
K_Dtheta   = 0;

%% Input for python script
% Initial conditions
x_ic        = 0;
theta_ic    = 30 * pi/180;
D_x_ic      = 0;
D_theta_ic  = 0;

IC = [x_ic, theta_ic, D_x_ic, D_theta_ic]';  
%% Input signals
dt = 0.001;
end_time = 6;

time = 0:dt:end_time;
u1 = ustep(time, 3);
u2 = ustep(time, 4);
u3 = ustep(time, 4) * 0.5;
u4 = ustep(time, 10) * 0.5;
u5 = ustep(time, 10) * -2;
u6 = ustep(time, 15) * -2;
ref_sig = u1 - u2 + u3 - u4 + u5 - u6;
ref_sig = ref_sig * 0;

% Structure with time; input to simulink model
U_sim.time = time;
U_sim.signals.values(:, 1) = ref_sig;
U_sim.signals.dimensions = 1;

%%
M = 0.05 ; mc = 0.1 ; mp = 0.1;
Lc = 0.3; Lp = 0.15;
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

%% Sim
simout = sim('simulation_model.slx');

%% Results
states  = simout.state_out.';
tout    = simout.tout.';
u       = simout.u.';

results.states = states;
results.tout   = tout;
results.u      = u;

%%
x       = states(1, :);
theta   = states(2, :);
Dx      = states(3, :);
Dtheta  = states(4, :);

%%
craneRailMin = -0.5;
craneRailMax = 0.5;

XLIM = [-0.75, 0.75];
YLIM = [-0.6, 0.6];

fig = figure();
ax = axes(fig);
ax.XLim = XLIM; ax.YLim = YLIM;
box(ax, 'on');
hold(ax, 'on');

craneRail = plot([craneRailMin, craneRailMax], [0, 0]);
craneRail.Color = 'Black';
craneRail.LineWidth = 1.5;

baseWidth = 0.12;
baseHeight = 0.07;
base = rectangle();
base.Position = [0-baseWidth/2, 0-baseHeight/2, baseWidth, baseHeight];
base.FaceColor = "#f5c33b";
base.EdgeColor = "black";
base.LineWidth = 1.5;
rotationAxis = plot(0, 0, 'o');
rotationAxis.MarkerFaceColor = 'black';
rotationAxis.MarkerEdgeColor = 'black';
base.addprop('rotationAxis');
base.rotationAxis = rotationAxis;

grid(ax, 'on');
axis equal;
% uistack(craneRail, 'bottom');

%% Oblicza współczynnik do decymacji stanu na podstawie FPS (frames per second)
FPS       = 30;
pauseTime = 1/FPS;
decimationFactor = pauseTime/dt;

drawnow();
pause(pauseTime);

x = x(1 : decimationFactor : end);

%% animacja
mov(1) = getframe;
for i = 2:length(x)
    base.Position(1) = x(i) - baseWidth/2;
    base.rotationAxis.XData = x(i);
    drawnow();
    pause(pauseTime);
    mov(i) = getframe;
end

% %% zapisywanie
% v = VideoWriter('animacja.avi');
% open(v);
% writeVideo(v, mov);
% close(v)



