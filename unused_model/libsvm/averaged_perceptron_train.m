function [averaged_w err] = averaged_perceptron_train(X, Y, update_step_fnc, numPasses)
% Trains an averaged perceptron on a sparse set of examples (X, Y)
%
% Example Usage:
%    update_constant_0_5 = @(x,y,w) update_constant(x,y,w,0.5)
%    [model err] = averaged_perceptron_train(Xtrain, Ytrain, @update_constant_0_5, 2)
%
% For a N x D sparse feature matrix X and Nx1 label matrix Y, returns
% averaged D x 1 weight model
% 
% For each example x_i, the weight vector will be updated by 
%           w = w + update_step_fnc(x_i,y_i,w)*x_i;
%
% numPasses is the number of times of passing the whole dataset through perceptron.
% The loop will end after number of passes (numPasses) is reached.
% 
% The function should also return an (NumPasses*N)x1 vector err containing the training
% error of each averaged weight vector. ??
% 

%For your convenience 
[N,D] = size(X);

%Initialize weights
w = 0.0001*ones(D,1);
%Keep a separate running sum of weights from each iteration
averaged_w = zeros(D,1);
total_w = zeros(D, 1);
numUpdates = 0;
err = zeros(numPasses*N,1);
%%YOUR CODE GOES HERE
for loopCount = 1:1:numPasses
    for iterator = 1:1:N
        w = w + update_step_fnc(X(iterator,:),Y(iterator,:),w)*X(iterator,:)';
        numUpdates = numUpdates + 1;
        total_w = total_w + w;
        averaged_w = total_w / numUpdates ;
        err((loopCount-1)*N + iterator, 1) = perceptron_error(X, Y, averaged_w);
    end
end

