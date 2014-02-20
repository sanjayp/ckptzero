function prediction = prediction_add_feature(model,test_words,test_meta)
% Take in a test_words, and a test_meta(original review text). The
% function would process the test_meta and extract ten punctuation
% information adding to the end of the test_words feature vector. Then it
% would make a prediction based on logistic regression.
%
% Input
% test_words : a 1xp vector representing "1" test sample.
% test_meta : a struct containing the metadata of the test sample.
% feature : adding additional punctuations ass new features(10). 
% model: use Logistic Regression
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

% Processing the test_meta
%punc list
puncs = ['.', ',', '!', ':' ,'(', '?', ';', '-'];
m = size(puncs,2);

str = test_meta.text;
len = size(str);

for j = 1:m
    punc_train(1,j) = sum(arrayfun(@(str) size(findstr(str{:},puncs(j)),2), str));
end
%'...'
punc_train(1,m+1) = sum(arrayfun(@(str) size(findstr(str{:},'...'),2), str));
punc_train(1,1) = punc_train(1,1) - 3 * punc_train(1,m+1);

%'!!'
punc_train(1,m+2) = sum(arrayfun(@(str) size(findstr(str{:},'!!'),2), str));
punc_train(1,3) = punc_train(1,3) - 2 * punc_train(1,m+2);

% make prediction
X = [test_words, punc_train];
[prediction, ~, ~] = predict(0, X, model.lg);
