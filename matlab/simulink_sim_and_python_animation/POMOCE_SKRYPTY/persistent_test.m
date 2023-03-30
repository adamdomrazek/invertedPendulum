%%
clc
% clear all
close all

%%
Ts = 0.001;
T_filtr = 0.1;
stop_time = 15;

%%
out = sim('persistent_test_m.slx');

%%
figure(1)
hold on
plot(out.tout, out.out.signals.values(:, 1), 'r--')
plot(out.tout, out.out.signals.values(:, 2), 'g-')
plot(out.tout, out.out.signals.values(:, 3), 'k--')
hold off
grid on
legend('sygnał szumiony sygnal', 'filtr ciagly', 'filtr dyskretny')


ax = gca;
ax.FontSize = 14;
ax.LineWidth = 1;
ax.GridAlpha = 0.5;

error = out.out.signals.values(:, 2) - out.out.signals.values(:, 3);
% figure(2)
% plot(out.tout, error)
