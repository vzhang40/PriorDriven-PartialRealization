function [V, W] = block_lanczos(A, R, L, n_max)
dtol = 1e-5;

% Block dimensions
m = size(R, 2);
p = size(L, 2);

%% Step 0: Initialization

% unnormalized starting vectors
Vhat = R;
What = L;

% Current left and right block sizes
m_c = m;
p_c = p;

% Records indices of vectors that must be preserved from inexact deflation
I_v = [];
I_w = [];

for n = 1:n_max

%% Step 1: deflate vhat_n if needed
    while norm(Vhat(:, n)) <= dtol
        disp(num2str(norm(Vhat(:, n))) + "=")
        disp("vhat_" + n +  " below tolerance, deflating")

        % Step 1a: check if block Krylov subspace is exhausted 
        if m_c == 1
            break 
        end

        % Step 1b: Saving index mu to preserve vectors v_mu and w_mu
        if n - m_c > 0 && any(Vhat(:, n))
            mu = n - m_c;
            I_w = [I_w, mu];
        end

        % Step 1c: vector vhat_n is deflated, indices are reduced
        for i = n : (n + m_c - 2)
            Vhat(:, i) = Vhat(:, i + 1);
        end
        m_c = m_c - 1; % reducing current right block size

        % Step 1d: repeat (1) if needed
    end

    if norm(Vhat(:, n)) <= dtol && m_c == 1 % Step 1a
        disp("Right block Krylov subspace exhausted, Stopping blockLanczos")
        disp("V and W truncated to size " + num2str(n-1))
        break
    end

%% Step 2: deflate what_n if needed
    while norm(What(:, n)) <= dtol
        disp(num2str(norm(What(:, n))) + "=")
        disp("what_" + n +  " below tolerance, deflating")

        % Step 2a: Check if block Krylov subspace is exhausted
        if p_c == 1
            break 
        end

        % Step 2b: Saving index phi to preserve vectors v_phi and w_phi
        if n - p_c > 0 && any(What(:, n))
            disp("Step 2b: phi > 0")
            phi = n - p_c;
            I_v = [I_v, phi];
        end

        % Step 2c: vector what_n is deflated, increase index
        for i = n : n+p_c-1
            What(:, i) = What(:, i + 1);
        end
        p_c = p_c - 1; % reducing left block size

        % Step 2d: repeat (2) if needed
    end

    if norm(What(:, n)) <= dtol && p_c == 1 % Step 2a
        disp("Left block Krylov subspace exhausted, Stopping blockLanczos")
        disp("V and W truncated to size " + num2str(n-1))
        break
    end

%% Step 3: Normalizing vhat_n and what_n
    mu = n - m_c;
    phi = n - p_c;
    if mu < 1 && phi < 1 % non-positive indices handled separately
        V(:, n) = Vhat(:, n) / norm(Vhat(:, n));
        W(:, n) = What(:, n) / norm(What(:, n));
    elseif mu < 1  % non-positive indices handled separately
        V(:, n) = Vhat(:, n) / norm(Vhat(:, n));
        t_tilde(n, phi) = norm(What(:, n));
        W(:, n) = What(:, n) / t_tilde(n, phi);
    elseif phi < 1
        t(n, mu) = norm(Vhat(:, n));
        V(:, n) = Vhat(:, n) / t(n, mu);
        W(:, n) = What(:, n) / norm(What(:, n));
    else
        t(n, mu) = norm(Vhat(:, n));
        t_tilde(n, phi) = norm(What(:, n));
        V(:, n) = Vhat(:, n) / t(n, mu);
        W(:, n) = What(:, n) / t_tilde(n, phi);
    end

%% Step 4: Advance the right block Krylov subspace 

    % Step 4a: get vector v
    v = A*V(:, n);

    % Step 4b: biorthogonalizing v against the other Lanczos vectors
    %   including the saved indices
    i_v = max([1, n-p_c]);
    if i_v <= n-1
        I = [I_v, i_v:n-1];
    else
        I = [I_v];
    end
    I = sort(I, "ascend");

    for k = 1:length(I)
        i = I(k);
        if i == n-p_c
            t(i, n) = t_tilde(n, i)*(W(:, n)'*V(:, n)) / (W(:, i)'*V(:, i));
        else 
            t(i, n) = (W(:, i)'*v) / (W(:, i)'*V(:, i));
        end
        v = v - V(:, i)* t(i, n);
    end
    Vhat(:, n+m_c) = v;

%% Step 5: Advance the left black Krylov subspace

    % Step 5a: get vector w
    w = A'*W(:, n);

    % Step 5b biorthogonalize w against previous lanczos vectors
    i_w = max([1, n-m_c]);
    if i_w <= n-1
        I = [I_w, i_w:n-1];
    else
        I = [I_w];
    end
    I = sort(I, "ascend");

    for k = 1:length(I)
        i = I(k);
        if i == n-m_c
            t_tilde(i, n) = t(n, i)*(W(:, n)'*V(:, n)) / (W(:, i)'*V(:, i));
        else
            t_tilde(i, n) = (w'*V(:, i)) / (W(:, i)'*V(:, i));
        end
        w = w - W(:, i)* t_tilde(i, n);
    end
    What(:, n+p_c) = w;

%% Step 6: Compute delta_n and check for breakdown and update auxiliary vectors

    % Step 6a: compute delta_n
    delta = W(:, n)'*V(:, n); 
    disp("delta = " + num2str(delta))

    % Step 6b: look-ahead needed to continue algorithm
    if norm(delta) < 1e-16
        disp("look ahead algorithm needed")
        W = W(:, 1:end-1);
        V = V(:, 1:end-1);
        disp("truncating at " + num2str(n-1))
        break
    end

    % Step 6c: Biorthogonalize vhat against w_n
    for i = n - m_c + 1:n
        if i <= 0
            Vhat(:, m_c + i) = Vhat(:, m_c + i) - V(:, n)*(W(:, n)'*Vhat(:, m_c + i)) / (W(:, n)'*V(:, n));
        elseif i == n
            t(n, i) = (W(:, n)'*Vhat(:, m_c + i)) / (W(:, n)'*V(:, n));
            Vhat(:, m_c + i) = Vhat(:, m_c + i) - V(:, n)*t(n, i);
        else
            t(n, i) = t_tilde(i, n)*(W(:, i)'*V(:, i)) / (W(:, n)'*V(:, n));
            Vhat(:, m_c + i) = Vhat(:, m_c + i) - V(:, n)*t(n, i);
        end
    end
    
    % Step 6d: Biorthogonalize what against v_n
    for i = n - p_c + 1:n
        if i <= 0
            What(:, p_c + i) = What(:, p_c + i) - W(:, n)*(What(:, p_c + i)'*V(:, n)) / (W(:, n)'*V(:, n));
        else
            t_tilde(n, i) = t(i, n)*(W(:, i)'*V(:, i)) / (W(:, n)'*V(:, n));
            What(:, p_c + i) = What(:, p_c + i) - W(:, n)*t_tilde(n, i);
        end
    end

    % Step 7: set up rho and eta
    if n <= m_c
        for i = n-m_c+m: m
            if i-m > 0
                rho(n, i) = t(n, i-m);
            else
                rho(n, i) = W(:, n)'*Vhat(:, m_c + i - m) / (W(:, n)'*V(:, n));
            end
        end
        m_1 = m_c;
    end
    if n <= p_c
        for i = n-p_c+p:p
            if i-p > 0
                eta(n, i) = t_tilde(n, i-p);
            else
                eta(n, i) = What(:, p_c + i -p)'*V(:, n) / (W(:, n)'*V(:, n));
            end
        end
        p_1 = p_c;
    end
end

% EXTRA: Step 8, making biorthonormal
D = diag(W'*V);
Dinv = diag(1./D);
W = W*Dinv;
end