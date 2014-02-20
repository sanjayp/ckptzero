function K = kernel_gaussian(X, X2, sigma)
% Evaluates the Gaussian Kernel with specified sigma
%
% Usage:
%
%    K = KERNEL_GAUSSIAN(X, X2, SIGMA)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the Guassian kernel
% with parameter sigma=20.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.
% vector_a * vector_a
asq = sum(X'.^2, 1)'; %column vector
% vector_a * vector_a - 2 * vector_a * vector_b
asq_2ab = bsxfun(@minus, asq, 2*X*X2');
% vector_a * vector_a - 2 * vector_a * vector_b + vector_b * vector_b
asq_2ab_bsq = bsxfun(@plus, sum(X2'.^2, 1), asq_2ab);
% exp( .../2sigma)
K = exp(-asq_2ab_bsq./(2*sigma^2));
K = K';


