% This is a script to create plots for PD-PR for the 1D advection-diffusion PDE
% example.
% 
% Copyright (c) 2026, Vivian Zhang
% All rights reserved.
% License: BSD 3-Clause License (see LICENSE)
%

clear; close all
rng(1, "twister")
addpath("functions")

%% Example Set-up
% set scale = 0 if no scaling
scale = 1; % creates an artificial scaled system with largest eigenvalue modulus == scale
T_end = 1; % ending times [1, 10, 50]

% Tested Reduced Dimensions
r_max = 20; % maximum dimension

% Full model Set-Up
d = 128; % spatial dimension
a = 1; % diffusion
c = [1, 100, 10000]; % advection

% Experiment Set up
n = 10; % number of timesteps
dt = T_end./n; % timestep size
r_vals = 1:r_max;
reps = 500;  % number of initial conditions drawn for posterior estimation

%% Initializing
HSV = zeros(length(c), d);

mu_errs_BT_hsv = zeros(length(c), length(r_vals));
mu_errs_PR_hsv = zeros(length(c), length(r_vals));

Gamma_errs_BT_hsv = zeros(length(c), length(r_vals));
Gamma_errs_PR_hsv = zeros(length(c), length(r_vals));

for i = 1:length(c)
[A, C] = get_matrices(d, a, c(i)); % state matrix and output matrix
Gamma_ep = 0.15^2; % noise covariance
d_out = size(C, 1); % output dimension

disp("dt = " + dt)
disp("T_end = " + T_end)
disp("max eig = " + max(abs(eig(A))))

%% Scaling: creates an artificial system such that system poles are in left unit circle
if scale ~= 0
    alpha = max(abs(eig(A))) + 1e-3;
    A = A./alpha * scale;
    C = full(C)./sqrt(alpha) * scale;
end
disp("new max eig = " + max(abs(eig(A))))

%% Obtaining prior
% given full low-rank prior - sample covariance of compatible covariance
ensemble_size       = 200; % size of prior ensemble
Lyap_solution       = lyapchol(A, eye(d))'; % compatibility Lyapunov eq.
ensemble            = Lyap_solution * randn(d, ensemble_size); 
Gamma_pr       = cov(ensemble'); % prior is ensemble covariance
prior_rank          = rank(Gamma_pr); % rank of prior covariance
[X, D]              = eig(Gamma_pr); % eigendecomposition of prior
[~, ind]            = sort(diag(D), 'descend'); % get important directions
Ds                  = D(ind, ind);
Xs                  = X(:, ind);
L_pr           = Xs(:, 1:prior_rank) ...
                        * sqrt(Ds(1:prior_rank, 1:prior_rank));

disp("Columns of L_pr: " + size(L_pr, 2))
disp("Rank of Gamma_pr: " + rank(Gamma_pr))

%% Full model and Measurements 
G = get_forward_model(A, C, dt, n);
p = L_pr * randn(size(L_pr, 2), reps);
Gamma_obs = kron(eye(n), Gamma_ep);
L_ep = chol(Gamma_ep, "lower");
epsilon = L_ep*randn(d_out, n*reps); % noise
epsilon = reshape(epsilon, [d_out*n, reps]);
y = G * p;
m = y + epsilon;

%% compute true posterior
[mu_pos, Gamma_pos] = compute_posterior(Gamma_pr, G, Gamma_obs, m);

%% Prior-Driven Balanced Truncation
Rp = lyapchol(A, L_pr)'; % Reachability Gramian
Lq = lyapchol(A', (Gamma_ep^(-1/2)*C)')'; % Observability Gramian
P_bt = Rp*Rp';
Q_bt = Lq*Lq';

% Balancing Transformation
[U, D_bt, Z] = svd(Lq'*Rp);
D_sqrtinv = diag(sqrt(1 ./ diag(D_bt)));
T = Rp*Z(:, 1:size(D_sqrtinv, 1))*D_sqrtinv;
S = Lq*U(:, 1:size(D_sqrtinv, 1))*D_sqrtinv;

ds = diag(D_bt);
HSV(i, :) = ds./sum(ds);

%% Prior-Driven Partial Realization
[V, W] = block_lanczos(A, L_pr, (Gamma_ep^(-1/2)*C)', r_max);

%% Getting Reduced Models

% Looping over basis sizes
for rr = 1:length(r_vals)
     r = r_vals(rr);

    %% Two-Sided Partial Realization
    Vr = V(:, 1:r);
    Wr = W(:, 1:r);

    % Reduced System
    A_pr = Wr'*A*Vr;
    Lpr_pr = Wr'*L_pr;
    C_pr = C*Vr;

    % Reduced forward model
    Gr = get_forward_model(A_pr, C_pr, dt, n);
    G_pr = Gr*Wr';

    % compute posterior estimates
    [mu_r_pos, Gamma_r_pos] = compute_posterior(Gamma_pr, G_pr, Gamma_obs, m);

    % Posterior Error
    Gamma_errs_PR_hsv(i, rr) = norm(Gamma_r_pos - Gamma_pos, 'fro') / norm(Gamma_pos, 'fro');
    mu_errs_PR_hsv(i, rr) =  mean(sum((mu_pos-mu_r_pos) .^ 2));

    %% Balanced Truncation
    Tr = T(:, 1:r); 
    Sr = S(:, 1:r);

    % Reduced System
    A_bt = Sr'*A*Tr;
    Lpr_bt = Sr'*L_pr;
    C_bt = C*Tr;

    % Reduced forward model
    Gr = get_forward_model(A_bt, C_bt, dt, n);
    G_bt = Gr*Sr';

    % compute posterior estimates
    [mu_r_pos, Gamma_r_pos] = compute_posterior(Gamma_pr, G_bt, Gamma_obs, m);

    % Posterior Error
    Gamma_errs_BT_hsv(i, rr) = norm(Gamma_r_pos - Gamma_pos, 'fro') / norm(Gamma_pos, 'fro');
    mu_errs_BT_hsv(i, rr) =  mean(sum((mu_pos-mu_r_pos) .^ 2));
end
end

%% Plots
legendStrings = [arrayfun(@(c) sprintf('$c/a = %d$', c), c, ...
    'UniformOutput', false)];

Gamma_errs_PR_hsv(Gamma_errs_PR_hsv >  1e10) = nan;
mu_errs_PR_hsv(mu_errs_PR_hsv >  1e10) = nan;


% Normalized Hankel Singular Values
figure(1); clf(1)
semilogy(1:size(HSV, 2), HSV(1, :), "o")
hold on
semilogy(1:size(HSV, 2), HSV(2, :), "x")
semilogy(1:size(HSV, 2), HSV(3, :), "sq")
xlabel('Index ($i$)', 'Interpreter', 'latex','fontsize', 18)
ylabel('$\delta_i$ - Normalized', 'Interpreter', 'latex','fontsize',18)
title('Norm. Hankel Singular Values', 'Interpreter', 'latex', 'fontsize', 20)
legend(legendStrings, 'Location', 'southwest' ,'fontsize', 18, 'Interpreter', 'latex')
xlim([1, d])
legend box off
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
grid on

figure(3); clf(3)
subplot(2, 3, 1)
semilogy(r_vals, Gamma_errs_BT_hsv(1, :), "o-") 
hold on
semilogy(r_vals, Gamma_errs_PR_hsv(1, :), "o:") 
hold on
ylabel('$\Gamma_{pos}$ - Rel. Frob. Error', 'Interpreter', 'latex','fontsize',18)
title('Adv-Diff Ratio   $c/a = 1$', 'Interpreter', 'latex', 'fontsize', 20)
legend('PD-BT', 'PD-PR', 'Location', 'southwest' ,'fontsize', 18, 'Interpreter', 'latex')
legend box off
xlim([min(r_vals), max(r_vals)])
grid on
miny = min([min(min(Gamma_errs_PR_hsv)), min(min(Gamma_errs_BT_hsv))]);
maxy = max([max(max(Gamma_errs_PR_hsv)), max(max(Gamma_errs_BT_hsv))]);
ylim([miny, maxy])
pbaspect([16 9 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')

subplot(2, 3, 2)
semilogy(r_vals, Gamma_errs_BT_hsv(2, :), "x-") 
hold on
semilogy(r_vals, Gamma_errs_PR_hsv(2, :), "x:") 
hold on
title('Adv-Diff Ratio   $c/a = 100$', 'Interpreter', 'latex', 'fontsize', 20)
legend('PD-BT', 'PD-PR', 'Location', 'southwest'  ,'fontsize', 18, 'Interpreter', 'latex')
legend box off
xlim([min(r_vals), max(r_vals)])
grid on
ylim([miny, maxy])
pbaspect([16 9 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')

subplot(2, 3, 3)
semilogy(r_vals, Gamma_errs_BT_hsv(3, :), "sq-") 
hold on
semilogy(r_vals, Gamma_errs_PR_hsv(3, :), "sq:") 
hold on
title('Adv-Diff Ratio   $c/a = 10000$', 'Interpreter', 'latex', 'fontsize', 20)
legend('PD-BT', 'PD-PR', 'Location', 'southwest' ,'fontsize', 18, 'Interpreter', 'latex')
legend box off
xlim([min(r_vals), max(r_vals)])
grid on
ylim([miny, maxy])
pbaspect([16 9 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')


subplot(2, 3, 4)
semilogy(r_vals, mu_errs_BT_hsv(1, :), "o-") 
hold on
semilogy(r_vals, mu_errs_PR_hsv(1, :), "o:") 
hold on
ylabel('$\mu_{pos}$ - Rel. MSE', 'Interpreter', 'latex','fontsize',18)
legend('PD-BT', 'PD-PR', 'Location', 'southwest'  ,'fontsize', 18, 'Interpreter', 'latex')
legend box off
xlim([min(r_vals), max(r_vals)])
grid on
miny = min([min(min(mu_errs_PR_hsv)), min(min(mu_errs_BT_hsv))]);
maxy = max([max(max(mu_errs_PR_hsv)), max(max(mu_errs_BT_hsv))]);
ylim([miny, maxy])
pbaspect([16 9 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')

subplot(2, 3, 5)
semilogy(r_vals, mu_errs_BT_hsv(2, :), "x-") 
hold on
semilogy(r_vals, mu_errs_PR_hsv(2, :), "x:") 
hold on
legend('PD-BT', 'PD-PR', 'Location', 'southwest' ,'fontsize', 18, 'Interpreter', 'latex')
legend box off
xlim([min(r_vals), max(r_vals)])
grid on
ylim([miny, maxy])
pbaspect([16 9 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')

subplot(2, 3, 6)
semilogy(r_vals, mu_errs_BT_hsv(3, :), "sq-") 
hold on
semilogy(r_vals, mu_errs_PR_hsv(3, :), "sq:") 
hold on
legend('PD-BT', 'PD-PR', 'Location', 'southwest' ,'fontsize', 18, 'Interpreter', 'latex')
legend box off
xlim([min(r_vals), max(r_vals)])
grid on
ylim([miny, maxy])
pbaspect([16 9 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')

%% Functions
function [A, C] = get_matrices(N, a, c)
% 1D advection-diffusion finite difference on unit domain
% N - spatial dimension
% a - diffusion coefficient
% c - advection coefficient
    dx = 1./N;
    A1 = diag(-2*ones(1, N)) + diag(ones(1, N-1), -1) + diag(ones(1, N-1), 1);
    A2 = diag(-1*ones(1, N-1), -1) + diag(ones(1, N-1), 1);
    A = (a / dx^2) * A1 - (c / (2*dx)) * A2;
    C = ones(1, N)./N;
end

function G = get_forward_model(A, C, dt, n)
% Given a zero-control system with the number of timesteps and step size, construct a
% forward model.
    [d_out, d] = size(C);
    G = zeros(d_out*n, d);
    temp = C;
    iter = expm(A*dt);
    for i = 1:n
        temp = temp * iter;
        G((i-1)*d_out + 1: i*d_out, :) = temp;
    end
end

function [mu_pos, Gamma_pos] = compute_posterior(Gamma_pr, G, Gamma_obs, m)
% Computes posterior quantities
    Gamma_pos = Gamma_pr - Gamma_pr * G' * ((Gamma_obs + G * Gamma_pr * G')\(G * Gamma_pr));
    mu_pos = Gamma_pos * G' * ((Gamma_obs) \ m);
end

