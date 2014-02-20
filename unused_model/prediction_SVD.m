function prediction = prediction_SVD(model,test_words,test_meta)
% The function take in a test_words, and do a transformation based on the
% SVD matrix then use logistic regression to make prediciton.
% Input
% test_words : a 1xp vector representing "1" test sample.
% test_meta : a struct containing the metadata of the test sample.
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
% 
% **Note: the function will only take 1 sample each time.


X = test_words(:, model.high_freq);
X = X * model.V0;
% Note: need to sparse the data
X = sparse(X);
[prediction, ~, ~] = predict(0, X, model.lg); % test the training data
