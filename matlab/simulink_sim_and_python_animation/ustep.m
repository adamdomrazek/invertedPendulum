function y = ustep(t, d)
% t: czas start support'u
% d : delay
% dla x(t - d)
N = length(t);
y = zeros(1,N);
for i = 1:N
    if t(i) >= d
        y(i) = 1;
    end
end
end