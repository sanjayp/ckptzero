function [X Y] = make_sparse(data, vocab)
% Returns a sparse matrix representation of the data.
%
% Usage:
%
%  [X, Y] = MAKE_SPARSE(DATA, VOCAB)
%
% For a struct array of newsgroup examples DATA, and a cell array
% vocabulary VOCAB, returns a sparse matrix X where X(i,j) is the # of
% times that word j occured in example i. Note that since X is sparse, only
% non-zero entries are stored. Also returns a binary label vector Y for the
% given data.

% Strip out -1's (unknown words in test set) from the counts.
for i = 1:numel(data)
    data(i).counts = data(i).counts(~[data(i).counts(:,1)==-1], :);
end

% YOUR CODE GOES HERE. Your job is to determine in rowidx, colidx, and values
% for the sparse matrix. If D is the number of NON ZERO values of X, then
% these are each D x 1 vectors. The idea here is that Matlab will create a
% sparse matrix data structure such that:
%                  X(rowidx(i),colidx(i)) = values(i).
% For more information about sparse matrices, see doc sparse.
%
% P.S., if we didn't use a sparse matrix, our full X matrix would take up
% 500 MB of memory!
% Step0: 

totalRows = size(data, 2);  % number of total examples(datasets)
totalColumns = length(vocab);
totalElements = totalRows * totalColumns;   %total number of possible elements in matrix X
% rowidx
% colidx
%colSequence = linspace(1, totalColumns, totalColumns);
%colMatrix = repmat(colSequence, totalRows, 1);
% colidx = reshape(colMatrix, totalElements, 1);
colidx = 0;
rowidx = 0;
values = 0;
for example = 1: 1: totalRows
    countsMatrix = getfield(data(example), 'counts');
    nonZeroElements = size(countsMatrix, 1);
    rowidx = vertcat(rowidx, ones(nonZeroElements, 1) * example);   %
    colidx = vertcat(colidx, countsMatrix(:,1));
    values = vertcat(values, countsMatrix(:,2));
    %disp(values);
end
% values
colidx = colidx(2:end, 1);
rowidx = rowidx(2:end, 1);
values = values(2:end, 1);
X = sparse(rowidx, colidx, values, numel(data), numel(vocab));

% Do not touch this: this computes the text label to a numeric 0-1 label,
% where 1 examples are mac newsgroup postings.
Y = double(cellfun(@(x)isequal(x, 'comp.sys.mac.hardware'), {data.label})');
