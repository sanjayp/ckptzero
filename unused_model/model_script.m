
addpath(genpath('./liblinear'));
clear_flag = 1;
%% Clear
if clear_flag
    clear all;
end
%% Flag
% adding additional features flag
review_length_flag = 1;
add_punc_flag = 1;
additional_feature_flag = 1;
% features
freq_flag = 0; 
% feature selection flag

porterStemmer_flag = 0;
stemmed_X_flag = 0;
stopwords_process_flag = 0;
stopwords_use_flag = 0;
idf_flag = 0;
SVDs_flag = 0;

% preprocessing
scale_flag = 0;
standardization_flag = 0;
normalization_flag = 0;

% models flag
SVM_flag = 0;   % too slow, some problem?
SVM_liblinear_flag = 1;
LG_flag = 0;
NB_flag = 0;
KNN_flag = 0;   % to be implemented
discriminant_flag = 0;
kmeans_flag = 0;

% To be implemented
neural_flag = 0;


% development mode
disp_flag = 1;
% if you want to leave some data for test, set to 1; if you want to get
% quiz result set to 0
train_verify_flag = 1;

%% Load Data
load ../data/review_dataset.mat;	% 
load ../data/add_features_quiz.mat;	% additional punctuation feature matrix: 5000*10
load ../data/add_features_train.mat;	% additional punctuation feature: 25000*10
load ../data/X_stemmed.mat;	% stemmed dataset 
load ../data/stopwords.mat;	% stopwords
load ../data/stopwords_ind.mat;     % including non_stopwords_ind
load ../data/non_stopwords_ind.mat; 
% var: punc_quiz(full), punc_train(full);
load ../data/punc.mat;  
% reset train data name to train2 to avoid confliction with func train
train2 = train;
clearvars train;


%% Adding features
if add_punc_flag
    train2.counts = [train2.counts, sparse(punc_train)];
end


%% set ind
data_comb = 3;

switch data_comb
    case 0
        disp('Data comb 0:');
        train_range_start = 1;
        train_range_end = 5000;
        test_range_start = 24001;
        test_range_end = 25000;        
    case 1
        disp('Data comb 1:');
        train_range_start = 1;
        train_range_end = 10000;
        test_range_start = 20001;
        test_range_end = 25000;  
    case 2
        disp('Data Comb 2:');
        train_range_start = 1;
        train_range_end = 20000;
        test_range_start = 20001;
        test_range_end = 25000;  
     case 3
        disp('Data Comb 3:');
        train_range_start = 1;
        train_range_end = 24000;
        test_range_start = 24001;
        test_range_end = 25000; 
    case 4
        disp('Data Comb 4:');
        train_range_start = 1;
        train_range_end = 25000;
        test_range_start = 20001;
        test_range_end = 25000;  
    otherwise
        disp('Default:');
end

%% Default 25000 * 65000
if disp_flag
    disp('Start Initializing Default Features:');
end
    if additional_feature_flag 
        train2.counts = sparse([train2.counts, sparse(add_features_train(:,1:3))]);
        %Xtest = [train2.counts,sparse(add_features_quiz)];
    end
    % pick number of observations to be tested
    
    %Xtest = [Xtest, sparse(punc_quiz)];
    if train_verify_flag
        X = train2.counts(train_range_start:train_range_end,:);      %train data 
        Y = train2.labels(train_range_start:train_range_end,:);       %train labels
        disp(size(X));
        disp(size(Y));

        Xtest = train2.counts(test_range_start:test_range_end,:);   %test data
        Ytest = train2.labels(test_range_start:test_range_end,:);    %test label
        
        
        disp(size(Xtest));
        disp(size(Ytest));
    else
        X = train2.counts;      %train data 
        Y = train2.labels;       %train labels
        Xtest = quiz.counts;   %test data
        Ytest = ones(size(quiz.counts, 1), 1);    %test label        
    end

    if disp_flag
        disp('    finished.');
    end
        
    
    
%% Stopwords
% find the index

if stopwords_process_flag
    tic;
    if disp_flag
        disp('Start Processing Stop Words:');
    end
    stopwords_i = 1;
    non_stopwords_i = 1;
    for iterator = 1:1:length(vocab)
        % if the word need to be stemmed
        find_stopword = 0;
        for iterator_stopwords = 1: 1: length(stopwords)
            if strcmp(vocab{iterator}, stopwords{iterator_stopwords})
                stopwords_ind(stopwords_i) = iterator;
                stopwords_i = stopwords_i + 1;
                find_stopword = 1;
            end
        end
        if ~find_stopword
            non_stopwords_ind(non_stopwords_i) = iterator;
            non_stopwords_i = non_stopwords_i + 1;            
        end
    end
    if disp_flag
        disp('    finished.');
    end 
    toc;

end    %end of the flag

if stopwords_use_flag
    X_extracted = train2.counts;
    temp = 1:1:size(vocab, 2);
    temp2 = stopwords_ind(:, 1:100 );
    temp(:, temp2) = [];
    temp3 = temp;
    disp('temp3');
    disp(size(temp3));
    X_extracted = X_extracted(:, temp3);
    X = X_extracted(train_range_start:train_range_end, :);
    Y = train2.labels(train_range_start:train_range_end,:);  
    Xtest = X_extracted(test_range_start:test_range_end,:);
    Ytest = train2.labels(test_range_start:test_range_end,:); 
    disp('    Note: Stop words used.');
end     %end of stopwords_use_flag


%% Use Stemmer

if stemmed_X_flag
    if exist('X_stemmed')
        X = X_stemmed;
    end
    ind = find(sum(X_stemmed) ~= 0);

    X_stemmed = X_stemmed(:, ind);
    disp(size(X_stemmed));

    X = X_stemmed(train_range_start:train_range_end, :);
    Y = train2.labels(train_range_start:train_range_end,:);  
    Xtest = X_stemmed(test_range_start:test_range_end,:);
    Ytest = train2.labels(test_range_start:test_range_end,:); 
    disp('    Note: Stemmer used.');
end

    
%% porter stemmer

if porterStemmer_flag
    tic;
    if disp_flag
        disp('Start Porter Stemmer:');
    end
    stem_map = zeros(1, length(vocab));
    stem_i = 0;
    X2 = X;
    for iterator = 1:1:length(vocab)

        if strcmp(vocab{iterator}, 'aed')
            continue;
        end
        % if the word need to be stemmed
        if ~strcmp(porterStemmer(vocab{iterator}), vocab{iterator})
            stemmed_to_ind = find(ismember(vocab, porterStemmer(vocab{iterator}) ));
            if ~isempty(stemmed_to_ind)
                stem_i = stem_i +1;
                stem_map(iterator) = stemmed_to_ind;
                X(:, stemmed_to_ind) = X(:, stemmed_to_ind) + X(:, iterator);
                X(:, iterator) = 0; % delete or just set to zero?
                if 0
                    disp(stem_i);
                    disp('--------');
                    disp(vocab{iterator});
                    disp(vocab{stemmed_to_ind});
                end
            end
        end
    end
    if disp_flag
        disp(stem_i);
        disp('    finished.');
    end 
    toc;
end    %end of the flag


    
%% additional feature

% add review length
if review_length_flag
    if disp_flag
       disp('Start adding review length as a feature.');
    end
    X = [X, sum(X, 2)];
    disp(size(X));
    Xtest = [Xtest, sum(Xtest, 2)];
    if disp_flag
        disp('    finished.');
    end
end




%% use word frequency

if freq_flag
    if disp_flag
        disp('Start Word Frequency Selection:');
    end
    total = cat(1, train2.counts, quiz.counts);
    freq = sum(total);
    RMSE_ind = 0;

    for f = 100
        RMSE_ind = RMSE_ind + 1;

        high_freq = find(freq>=f);  %find high frequency count
        low_freq = find(freq<f);    %find low frequency count
        % Note: use X here, not train2.counts
        %highfreq_count = X(:,high_freq);    
        lowfreq_count = X(:, low_freq);
        lowfreq_sum = sum(lowfreq_count')'; %sum up the low frequency as a feature

        X = X(:,high_freq);     %train data 
        Y = Y;       %train labels
        Xtest = Xtest(:, high_freq);   %test data
        Ytest = Ytest;    %test label    
    end
    if disp_flag
        disp(size(X));
        disp('   finished!');
    end
end


%% Scale

% not OK. after scale, the matrix becomes full

if scale_flag
    if disp_flag
        disp('Start Scaling:');
    end    
    [rows, columns]=size(X);   % A is your matrix
    % why abs?
    colMax = max(abs(X),[],1);    % take max absolute value to account for negative numbers
    colMin = min(abs(X), [], 1);    %all zero infact
    colRange = colMax - colMin;    % may contain zeros
    colMean = mean(X);
    % minus a mean(center)
    X = bsxfun(@minus, X, colMin);
    % what if some range contains zeros?
    X = bsxfun(@rdivide, X, colRange);
    
    
    colMax_test = max(abs(X),[],1);    % take max absolute value to account for negative numbers
    colMin_test = min(abs(X), [], 1);    %all zero infact
    colRange_test = colMax - colMin;    % may contain zeros

    % minus a mean(center)
    Xtest = bsxfun(@minus, Xtest, colMin_test);
    % what if some range contains zeros?
    Xtest = bsxfun(@rdivide, Xtest, colRange_test);
end

%% Standarzation

% not OK. after scale, the matrix becomes full

if standardization_flag
    if disp_flag
        disp('Start Standardization:');
    end
    
    col_mean = mean(X);
    col_std = std(X);
    X = bsxfun(@minus, X, col_mean);
    % what if some range contains zeros?
    X = bsxfun(@rdivide, X, col_std);
    
    % process Xtest in the same way
    col_mean_test = mean(Xtest);
    col_std_test = std(Xtest);
    Xtest = bsxfun(@minus, Xtest, col_mean_test);
    % what if some range contains zeros?
    Xtest = bsxfun(@rdivide, Xtest, col_std_test);
    
    if disp_flag
        disp('    fishined.');
    end
end


%% Normaliztion

% not OK. after scale, the matrix becomes full
% Note: normalize over row not column
if normalization_flag
    if disp_flag
        disp('Start Normalization:');
    end
    
    col_norm = sqrt(sum(X.^2, 2));
    X = bsxfun(@rdivide, X, col_norm);
    
    % process Xtest in the same way
    col_norm_test = sqrt(sum(Xtest.^2, 2));
    % what if some range contains zeros?
    Xtest = bsxfun(@rdivide, Xtest, col_norm_test);
    
    if disp_flag
        disp('    fishined.');
    end
end



%% Construct idf vector for features
% author: Sanjay

if idf_flag
    if disp_flag
        disp('Start IDF:');
    end
    if 0
        total_X = train2.counts;
    else
        total_X = [X ; Xtest];
    end
    num_features = size(total_X,2);
    df = zeros(1,num_features);
    d = zeros(1,num_features);
    d(1:num_features) = num_features;   %total
    for i = 1:num_features
        % Note: using add-1 smoothing
        df(1,i) = nnz(total_X(:,i)) + 1;  % number of non-zero counts
    end
    idf = log(d ./ df); % total counts/feature counts
    [sorted_idf, IDX] = sort(idf);
    idftrain = total_X(:,IDX);    % reorder the features according to IDF
    % smaller idf is more representative
    if 0
        for iterator = 1:10
            X = idftrain(train_range_start:train_range_end, 1:5000);   
            Xtest = idftrain(test_range_start:test_range_end, 1:5000);        
        end
    else
        X = [X , idftrain(train_range_start:train_range_end, 1:10000)];   
        Xtest = [Xtest, idftrain(test_range_start:test_range_end, 1:10000)];
    end
    if disp_flag
        disp('Updated feature size');
        disp(size(X));
        disp(size(Xtest));
        disp('    finished.');
    end    
end

%% SVDs(PCA)

if SVDs_flag
    if disp_flag
        disp('Start SVDs(fsvd):');
    end
    for components = 100  %components used
        fsvd_power = 2; %fsvd power, 2 as default
        % center the data(based on train or train+test?
        X_total = cat(1, X, Xtest);
        X0 = bsxfun(@minus, X, mean(X_total,1));
        X0test = bsxfun(@minus, Xtest, mean(X_total,1));
        X_total_centered = bsxfun(@minus, X_total, mean(X_total,1));
        [U0, S0, V0] = fsvd(X_total_centered, components, fsvd_power);
        %A0 = U0(:,1:k) * S0(1:k,1:k) * V0(:,1:k)';
        X_new = X0*V0;
        X_test_new = X0test*V0;
        X = sparse(X_new);
        Xtest = sparse(X_test_new);
    end
    if disp_flag
        disp('components used: ');
        disp(components);
        disp(size(X));
        disp(size(Xtest));
        disp(size(Ytest));
        disp('    finished!');
    end
end
%% MODELS

%% SVM

if SVM_flag
    if disp_flag
        disp('Start SVM:');
        disp(size(X));
        disp(size(Xtest));
    end
    if ~exist('kernel_gaussian')
        clearvars kernel_gaussian;  %need to clear repeat variable; otherwise error
    end
    kernel_gaussian = @(x,x2) kernel_gaussian(x, x2, 20);   % C = 100 for Gaussian
    kernel_intersect = @(x,x2) kernel_intersection(x, x2);
    tic;
    % kernel_libsvm modify to specify C. If you need to use cross
    % validation to determine C, modify back.
    % C = 100 for gaussian, C  = 0.01 for intersect
    addpath(genpath('./libsvm'));
    if 1
        [results_gaussian, info_gaussian] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_gaussian, 100);
        RMSE_SVM_gaussian = sqrt(norm(info_gaussian.yhat - Ytest, 2)^2 /length(Ytest));
    else
        [results_intersect, info_intersect] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_intersect, 0.01);% ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
        RMSE_SVM_intersect = sqrt(norm(info_intersect.yhat - Ytest, 2)^2 /length(Ytest))
    end
    toc;
    RMSE_SVM = sqrt(norm(info_gaussian.yhat - Ytest, 2)^2 /length(Ytest));
    disp(RMSE_SVM);
    if disp_flag
        disp('    SVM finished.');
    end
end


%% Logistic Regression
if LG_flag
    if disp_flag
        disp('Start Logistic Regression:');
    end

    tic;
    % best RMSE parameters: '-s 7 -c 0.06 -e 0.001 -q'
    model = train(Y, X, '-s 7 -c 0.06 -e 0.001 -q');
    [prediction, accuracy, dec_values] = predict(Ytest, Xtest, model); % test the training data
    RMSE892 = sqrt(norm(prediction - Ytest, 2)^2 /length(Ytest));
    disp(RMSE892);
    toc;
    if disp_flag
        disp('  finished.');
    end    
end

%% SVM from Liblinear
if SVM_liblinear_flag
    if disp_flag
        disp('Start SVM from Liblinear:');
    end
 
    tic;
    % best RMSE parameters: '-s 7 -c 0.06 -e 0.001 -q'
    %model = train(Y, X, '-s 1 -v 5 -e 0.01 -q');
    if 0
        crange = linspace(0.02, 0.001, 10);
        for i = 1:numel(crange)
            acc(i) = train(Y, X, sprintf('-s 5 -v 10 -c %g -e 0.001 -q', crange(i)));
        end
        [~, bestc_ind] = max(acc);
        fprintf('Cross-val chose best C = %g\n', crange(bestc_ind));  
        best_c = crange(bestc_ind);
    else
        % modify here to change parameter C
        best_c = 0.005
    end
    if 0
        crange = [1 2 3 4 5];
        for i = 1:numel(crange)
            acc(i) = train(Y, X, sprintf('-s %g -v 10 -c 0.005 -e 0.001 -q', crange(i)));
        end
        [~, bests_ind] = max(acc);
        fprintf('Cross-val chose best s = %g\n', crange(bests_ind));  
        best_s = crange(bests_ind);
    end
    if 1
        model = train(Y, X, sprintf('-s 1 -c %g -e 0.001 -q', best_c));
    else
        model = train(Y, X, sprintf('-s 3 -c 0.02 -e 0.001 -q'));
    end
    [prediction, accuracy, dec_values] = predict(Ytest, Xtest, model); % test the training data
    RMSE892 = sqrt(norm(prediction - Ytest, 2)^2 /length(Ytest));
    disp(RMSE892);
    toc;
    if disp_flag
        disp('  finished.');
    end    
end


%% Naive Bayes

% accuracy around 1.04

if NB_flag
    if disp_flag
        disp('Start Naive Bayes:');
    end  
    model_nb = NaiveBayes.fit(X, Y, 'Distribution', 'mn');

	prediction = model_nb.predict(Xtest);
    RMSE_nb = sqrt(norm(prediction - Ytest, 2)^2 /length(Ytest));
    disp(RMSE_nb);    
end

%% Discriminant Analysis

if discriminant_flag
    if disp_flag
        disp('Start Discriminant Analysis:');
    end  
    model_discriminant = ClassificationDiscriminant.fit(X, Y, 'discrimType','diagQuadratic');
    prediction = predict(model_discriminant, Xtest);
    RMSE_discriminant = sqrt(norm(prediction - Ytest, 2)^2 /length(Ytest));
    disp(RMSE_discriminant);
end


%% K-means
% data is unbalanced, we should try 
if kmeans_flag
    if disp_flag
       disp('Start K-means:');
    end
    total_X = [X; Xtest];
    total_Y = [Y; Ytest];
    


    opts = statset('MaxIter',500);
    prediction_Y = kmeans(total_X, 5, 'options', opts);   
    %prediction_Y = kmeans(total_X, 5);   
    prediction_Y2 = prediction_Y(1:size(X,1));
    %prediction = zeros(size((total_X), 1), 1);
    disp('kmeans');
    
    for iterator = 1:5
        disp(iterator);
        % get a cluster
        label_ind_train = find(prediction_Y2 == iterator);
        % get the cluster's real label
        prediction_cluster_train = Y(label_ind_train);
        % use majority label(Y) of this clusteras the true label
        x_cluster_label = mode(prediction_cluster_train);
        % set the cluster label to true label
        prediction(label_ind_train) = x_cluster_label;
    end

% 
%     for iterator = 1:5
%         disp(iterator);
%         % get a cluster
%         label_ind_train = find(prediction_Y2 == iterator);
%         % get the cluster's real label
%         prediction_cluster_train = Y(label_ind_train);
%         % use majority label(Y) of this clusteras the true label
%         x_cluster_label = mode(prediction_cluster_train);
%         % set the cluster label to true label
%         prediction(label_ind_train, 1) = x_cluster_label;
%     end
    if disp_flag
       disp('	finished.');
    end    
end
