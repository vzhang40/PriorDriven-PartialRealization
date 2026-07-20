function [V, W] =  pr_get_VW(A, B, C, r)

% This function determines matrices V and W which can be used to construct
% a ROM whose first 2r+1 moments at s = inf match the ones of the FOM [A,B,C,D]

% Initialization of V and W
n = length(A) ; % state space dimension
V = NaN(n, r) ;
W = NaN(n, r) ;

% The function apply_matrix is applied in each iteration to the
% previous column of V. If s0 is infinite, we need to perform a simple
% matrix multiplication from the left by A.
apply_matrix = @(v) A*v ;
% The function apply_adjoint is applied in each iteration to the
% previous column of W. If s0 is infinite, we need to perform a simple
% matrix multiplication from the left by A'.
apply_adjoint = @(v) A'*v ;
% Determine first columns of V and W (before normalization)
V(:,1) = B ;
W(:,1) = C' ;

% The columns of V and W are normalized such that the columns of V have 
% norm 1 and the columns of W are scaled such that W(:,i)'*V(:,i)=1 holds.
V(:,1) = V(:,1)/norm(V(:,1)) ;
W(:,1) = W(:,1)/(V(:,1)'*W(:,1)) ;

for k=2:r % loop over the remaining columns of V and W
    v_tilde = apply_matrix(V(:,k-1)) ; % new column vector for V
    w_tilde = apply_adjoint(W(:,k-1)) ; % new column vector for W
    % It is ensured that the V(:,k) is orthogonal to W(:,1:k-1) and W(:,k) 
    % is orthogonal to V(:,1:k-1). Moreover, the columns are scaled as 
    % mentioned above.
    V(:,k) = v_tilde-V(:,1:k-1)*(W(:,1:k-1)'*v_tilde) ;
    V(:,k) = V(:,k)/norm(V(:,k)) ;
    W(:,k) = w_tilde-W(:,1:k-1)*(V(:,1:k-1)'*w_tilde) ;
    W(:,k) = W(:,k)/(V(:,k)'*W(:,k)) ;
end