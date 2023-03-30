echo off
%
%   help groot
%   get(groot, 'factory'); 
%
%   Writes all extra settings to a file
%       diary dFile
%       get(groot, 'factory')
%       diary off
%%
clc

%% Change text interpreter to latex
list_factory = fieldnames(get(groot,'factory'));
index_interpreter = find(contains(list_factory,'Interpreter'));
for i = 1:length(index_interpreter)
    default_name = strrep(list_factory{index_interpreter(i)},'factory','default');
    set(groot, default_name,'latex');
end

%% Load data to plot
load('simout.mat')

%%
states = results.states;
tout   = results.tout;
u      = results.u;

x       = states(1, :);
theta   = states(2, :) * 180/pi;
Dx      = states(3, :);
Dtheta  = states(4, :) * 180/pi;

%% 

figure(1)
axis equal
set(gcf,'position',[500 200 1000 600])
set(gca,'FontWeight','bold')

sgt = sgtitle('\textbf{Wyniki symulacji}');

subplot(3, 2, 1)

plot(tout, x, 'k-')
ylabel('$x\ [ m ]$')
xlabel('t [sek]'); 
grid on;

subplot(3, 2, 2)
plot(tout, theta, 'k-')
ylabel('$\theta\ [ ^\circ ]$')
xlabel('t [sek]'); 
grid on;


subplot(3, 2, 3)
plot(tout, Dx, 'k-')
ylabel('$\dot{x}\ \left[ \frac{m}{s} \right]$')
xlabel('t [sek]'); 
grid on;

subplot(3, 2, 4)
plot(tout, Dtheta, 'k-')
ylabel('$\dot{\theta}\ \left[ \frac{^\circ}{s} \right]$')
xlabel('t [sek]'); 
grid on;

subplot(3, 2, 5)
plot(tout, u)
ylabel('u [N]')
xlabel('t [sek]'); 
grid on;

subplot(3, 2, 6)
plot(tout, u)
ylabel('u [N]')
xlabel('t [sek]'); 
grid on;

%% Rotate y label for all subplots
for i = 1:6
    subplot(3, 2, i)
    hYLabel = get(gca,'YLabel');
    set(hYLabel,'rotation',0,'HorizontalAlignment','right')
    
    ax = gca;
    ax.XAxis.LineWidth = 1;
    ax.YAxis.LineWidth = 1.4;
    
%     ax.XLabel.BackgroundColor = [0, 0, 1]
    ax.XLabel.FontWeight = 'bold';
end