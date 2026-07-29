% For debugging purposes, this function plots the state variable evolution
% over time with 
%   p - initial condition
%   A - state matrix
function plot_field(A, p)
    figure(1); clf(1)
    dt = 0.001;
    n = 10./dt;
    t = dt*1:dt:dt*n;
    d = size(A, 1);
    x = linspace(0, 1, d+2);
    x = x(2:end-1);
    G = get_forward_model(A, eye(d), dt, n);
    X = G*p;
    X = reshape(X, [d, n]);
    subplot(2, 1, 1)
    mesh(t, x, X)
    hold on
    axis equal
    xlabel('time')
    ylabel('space')
    title('1D Advection-Diffusion')
    subplot(2, 1, 2)
    imagesc(t, x, X)
    axis equal
end