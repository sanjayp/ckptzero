function prediction = make_final_prediction(model,test_words,test_meta)

% Input
% test_words : a 1xp vector representing "1" test sample.
% test_meta : a struct containing the metadata of the test sample.
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

% use liblinear's Logistic regression(-s 7) as model
[prediction, ~, ~] = predict(1, test_words, model.lg);
