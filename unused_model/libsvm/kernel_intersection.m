function K = kernel_intersection(X, X2)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    K = KERNEL_INTERSECTION(X, X2)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the histogram
% intersection kernel.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.
X_tranpose = X';
for iterator = 1:1:n
    VX_repmatrix = repmat((X_tranpose(:, iterator))', m, 1);
    K(:,iterator) = sum(min(VX_repmatrix, X2),2);    % m*1 vector
end
