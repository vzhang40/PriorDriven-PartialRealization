function [V, H] = arnoldi(A, b, r_max)
% computed arnoldi iteration
% V : its r_max columns are an orthonormal basis of the Krylov subspace 
% H : The matrix A on basis V. It is upper Hessenberg.
    if nargin < 3
        r_max = size(A, 1); % state dimension
    end
    H = zeros(r_max, r_max-1);
    V = zeros(size(A, 1), r_max);
    V(:, 1) = b / norm(b);
    for k = 2:r_max+1
        v = A*V(:, k-1); 
        for j = 1:k-1
            H(j, k-1) = V(:, j)'*v;
            v = v - H(j, k-1) * V(:, j);
        end
        if norm(v) < 1e-12 || k == r_max + 1
            break
        end
        H(k , k-1) = norm(v);
        V(:, k) = v / H(k, k-1);
    end
end
