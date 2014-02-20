tic;
load ../data/review_dataset.mat;

model = init_model_SVD(vocab);
Xtest = train.counts(15001:end, :);
Ytest = train.labels(15001:end, 1);
for iterator = 1:10000
    disp(iterator);
    test_words = Xtest(iterator, :);
    test_meta = 1;
    % prediction is a row vector
    prediction = prediction_SVD(model,test_words,test_meta);
    total_prediction(iterator, :) = prediction;
end 
% a vector
RMSE = sqrt(norm(total_prediction - Ytest, 2)^2 /length(Ytest));
toc;