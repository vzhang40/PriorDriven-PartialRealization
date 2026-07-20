clear; close all
rng(1, "twister")
addpath("functions")

%% Experimental Parameters

reps = 500;  % number of initial conditions drawn for posterior estimation
scale = true; % creates an artificial system such that system poles are in left unit circle

% Time Scale
T_end = 10; % ending time
dt = 0.5; % timestep

% Full model variables
d = 128; % spatial dimension
a = 0.1; % diffusion
c = 0.1; % advection

% Parameters
[A, C] = get_matrices(d, a, c);
Gamma_ep = 0.15^2;
d_out = size(C, 1);
r_max = 20;
r_vals = 1:r_max;

n = floor(T_end/dt); % number of timesteps
disp("dt = " + dt)
disp("T_end = " + T_end)
disp("max eig" + max(abs(eig(A))))

%% Scaling 
if scale == true
    alpha = max(abs(eig(A))) + 1e-5;
    A = A./alpha;
    % B = full(B)./sqrt(alpha);
    C = full(C)./sqrt(alpha);
end


%% given full low-rank prior - sample covariance of compatible covariance
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

%%
G = get_forward_model(A, C, dt, n);
p = L_pr * randn(size(L_pr, 2), reps);
Gamma_obs = kron(eye(n), Gamma_ep);
L_ep = chol(Gamma_ep, "lower");
epsilon = L_ep*randn(d_out, n*reps); % noise
epsilon = reshape(epsilon, [d_out*n, reps]);
y = G * p;
m = y + epsilon;

%% Visualizing PDE
plot_field(A, p(:, 1))

%% Full Prior-Driven Model
% Markov Parameters
N = 3*r_max;
H = markov_parameters(N, A, L_pr, Gamma_ep^(-1/2)*C);

% compute true posterior
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

%% Prior-Driven Partial Realization
[V, W] = block_lanczos(A, L_pr, (Gamma_ep^(-1/2)*C)', r_max);

%% Getting Reduced Models
moments_matched = zeros(length(r_vals), 2);
moment_diff_pr = zeros(length(r_vals), N);
moment_diff_bt = zeros(length(r_vals), N);

Gamma_err = zeros(length(r_vals), 2);
mu_mse = zeros(length(r_vals), 2);

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

    % Markov Parameter 
    H_pr = markov_parameters(N, A_pr, Lpr_pr, (Gamma_ep^(-1/2))*C_pr);
    [moments_matched(rr, 1), moment_diff_pr(rr, :)] = momentsMatched(H_pr, H);

    % Reduced forward model
    Gr = get_forward_model(A_pr, C_pr, dt, n);
    G_pr = Gr*Wr';

    % compute posterior estimates
    [mu_r_pos, Gamma_r_pos] = compute_posterior(Gamma_pr, G_pr, Gamma_obs, m);

    % Posterior Error
    Gamma_err(rr, 1) = norm(Gamma_r_pos - Gamma_pos, 'fro') / norm(Gamma_pos, 'fro');
    mu_mse(rr, 1) =  mean(sum((mu_pos-mu_r_pos) .^ 2));

    %% Balanced Truncation
    Tr = T(:, 1:r); 
    Sr = S(:, 1:r);

    % Reduced System
    A_bt = Sr'*A*Tr;
    Lpr_bt = Sr'*L_pr;
    C_bt = C*Tr;

    % Markov Parameter 
    H_bt = markov_parameters(N, A_bt, Lpr_bt, (Gamma_ep^(-1/2))*C_bt);
    [moments_matched(rr, 2), moment_diff_bt(rr, :)] = momentsMatched(H_bt, H);

    % Reduced forward model
    Gr = get_forward_model(A_bt, C_bt, dt, n);
    G_bt = Gr*Sr';

    % compute posterior estimates
    [mu_r_pos, Gamma_r_pos] = compute_posterior(Gamma_pr, G_bt, Gamma_obs, m);

    % Posterior Error
    Gamma_err(rr, 2) = norm(Gamma_r_pos - Gamma_pos, 'fro') / norm(Gamma_pos, 'fro');
    mu_mse(rr, 2) =  mean(sum((mu_pos-mu_r_pos) .^ 2));
end
%%
set(groot, 'DefaultAxesFontSize', 18)

% % Markov Parameter Matching
% figure(3); clf(3)
% subplot(2, 2, [1 2])
% plot(r_vals, moments_matched(:, 2), "o-")
% hold on
% plot(r_vals, moments_matched(:, 1), "sq:")
% plot(r_vals, floor(r_vals./d_out) + floor(r_vals./size(L_pr, 2)) + 1, "LineWidth", 0.5)
% plot(r_vals, floor(r_vals./d_out) + 1, "LineWidth", 0.5)
% ylabel('Moments Matched', 'Interpreter', 'latex')
% xlabel('Reduced Dimension', 'Interpreter', 'latex')
% title("Moments Matched tol $= 10^{-5}$", 'Interpreter', 'latex', 'fontsize', 20)
% legend('balanced truncation', 'partial realization (2)', 'lower bound two-sided matching', 'lower bound left matching', 'Location', 'northwest','fontsize',18)
% legend box off
% xlim([min(r_vals), max(r_vals)])
% pbaspect([16 4.5 1])
% 
% labels = [{'cut-off error'}, ...
%     compose('r = %d', round(r_vals))];
% 
% idx = find(any(isnan(moment_diff_bt),1), 1, 'first');
% if ~isempty(idx)
%     moment_diff_bt = moment_diff_bt(:,1:idx-1);
%     N = idx - 1;
% end
% 
% subplot(2, 2,3)
% imagesc([1e-5*ones(1, N); moment_diff_bt])
% hold on
% colorbar
% set(gca, 'ColorScale', 'log')
% colormap('pink')
% clim([1e-6, 1])
% yticks(1:length(labels))
% yticklabels(labels)
% xlabel('Leading Moments', 'Interpreter','latex')
% title('Moment Relative Error; Balanced Truncation', 'Interpreter','latex', 'fontsize', 20)
% 
% 
% idx = find(any(isnan(moment_diff_pr),1), 1, 'first');
% 
% pbaspect([16 9 1])
% 
% if ~isempty(idx)
%     moment_diff_pr = moment_diff_pr(:,1:idx-1);
%     N = idx - 1;
% end
% 
% subplot(2, 2, 4)
% imagesc([1e-5*ones(1, N); moment_diff_pr])
% hold on
% colorbar
% set(gca, 'ColorScale', 'log')
% colormap('pink')
% clim([1e-6, 1])
% yticks(1:length(labels))
% yticklabels(labels)
% xlabel('Leading Moments', 'Interpreter','latex')
% title('Moment Relative Error; Partial Realization (2)', 'Interpreter','latex', 'fontsize', 20)
% 
% pbaspect([16 9 1])

% posterior quantity errors
Gamma_err(Gamma_err > 1e10) = nan;
mu_mse(mu_mse > 1e10) = nan;

% Separate Plot for just posteriors
figure(15); clf(15)
subplot(2, 1, 1)
semilogy(r_vals, Gamma_err(:, 2), "o-") 
hold on
semilogy(r_vals, Gamma_err(:, 1), "sq:")
ylabel('Relative Frobenius Error', 'Interpreter', 'latex')
xlabel('Reduced Dimension', 'Interpreter', 'latex')
title("$\Gamma_{pos}$ Error", 'Interpreter', 'latex', 'fontsize', 20)
legend('balanced truncation', 'partial realization (2)', 'Location', 'southwest','fontsize',18)
legend box off
xlim([min(r_vals), max(r_vals)])
pbaspect([16 4.5 1])


subplot(2, 1, 2)
semilogy(r_vals, mu_mse(:, 2), "o-") 
hold on
semilogy(r_vals, mu_mse(:, 1), "sq:")
ylabel('Relative Mean Square Error', 'Interpreter', 'latex')
xlabel('Reduced Dimension', 'Interpreter', 'latex')
title("$\mu_{pos}$ Error", 'Interpreter', 'latex', 'fontsize', 20)
legend('balanced truncation', 'partial realization (2)', 'Location', 'southwest','fontsize',18)
legend box off
xlim([min(r_vals), max(r_vals)])
ax = gca;
ax.YScale = 'log';
ytickformat('10^{%g}')
pbaspect([16 4.5 1])

% Hankel Singular Values
figure(16); clf(16)
ds = diag(D_bt);
semilogy(ds, "o-")
hold on
xline(r_max, 'k-', 'Cutoff', 'LineWidth', 0.1);
ylabel('Hankel Singular Values', 'Interpreter', 'latex')
xlabel('index', 'Interpreter', 'latex')
title('Hankel Singular Values', 'Interpreter', 'latex', 'fontsize', 20)
pbaspect([16 9 1])

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

% plot field
function plot_field(A, p)
    figure(1); clf(1)
    dt = 0.001;
    n = 1./dt;
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
    pbaspect([1, 1, 1])
    xlabel('time')
    ylabel('space')
    title('1D Advection-Diffusion')
    subplot(2, 1, 2)
    imagesc(t, x, X)
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
% Computes posterior
    Gamma_pos = Gamma_pr - Gamma_pr * G' * ((Gamma_obs + G * Gamma_pr * G')\(G * Gamma_pr));
    mu_pos = Gamma_pos * G' * ((Gamma_obs) \ m);
end

function H = markov_parameters(N,A,B,C,D)
% This computes the first N markov parameters
H = cell(N,1);
if nargin < 5
    H{1} = zeros(size(C, 1), size(B, 2));
else
    H{1} = D;
end

Ak = eye(size(A, 1));

for k = 2:N
    H{k} = C * Ak * B;
    Ak = Ak * A;
end
end

function [num, moment_diff] = momentsMatched(Hr, H)
% given the markov parameters of two systems, this function computes the
% number of parameters matched with some tolerance 'ep' as well as the
% relative error in 'moment_diff'
ep = 1e-5;
N = min([length(Hr), length(H)]);
moment_match = false(1, N);
moment_diff = zeros(1, N);
for r = 1:N
    if any(isnan(H{r}))
        moment_diff(r) = nan;
    elseif norm(H{r}) == 0 
        moment_diff(r) = norm(Hr{r} - H{r});
        moment_match(r) = moment_diff(r) < ep; % can be generalized for MIMO
    else
        moment_diff(r) = norm(Hr{r} - H{r})./norm(H{r});
        moment_match(r) = moment_diff(r) < ep; % can be generalized for MIMO
    end
end
num = sum(moment_match);
end

