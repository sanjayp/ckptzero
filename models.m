


%% Clear

clear all;
%% Flag
freq_flag = 0;

porterStemmer_flag = 0;
stemmed_X_flag = 1;
idf_flag = 0;
SVDs_flag = 0;
% models flag
SVM_flag = 0;   % too slow, some problem?
LG_flag = 1;
NB_flag = 0;
KNN_flag = 0;
discriminant_flag = 0;

% To be implemented
neural_flag = 0;

% preprocessing
scale_flag = 0;
standardization_flag = 0;
normalization_flag = 0;

% features
additional_feature_flag = 0;

disp_flag = 1;
% if you want to leave some data for test, set to 1; if you want to get
% quiz result set to 0
train_verify_flag = 1;
%% Load Data

load ../data/review_dataset.mat;
load ../data/add_features_quiz.mat;
load ../data/add_features_train.mat;
load ../data/X_stemmed;

% reset train data name to train2 to avoid confliction with func train
train2 = train;
clearvars train;

%% set ind

data_comb = 3;

switch data_comb
    case 0
        disp('Data comb 0:');
        train_range_start = 1;
        train_range_end = 5000;
        test_range_start = 20001;
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
        disp('Data Comb 2:');
        train_range_start = 1;
        train_range_end = 24000;
        test_range_start = 24001;
        test_range_end = 25000; 
    case 4
        disp('Data Comb 3:');
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
        disp('Stemmer Used.');
    end
    if disp_flag
        disp('    finished.');
    end
        
    

%% additional feature






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

if idf_flag
    if disp_flag
        disp('Start IDF:');
    end
    num_feats = length(train2.counts);
    df = zeros(1,num_feats);
    d = zeros(1,num_feats);
    d(1:num_feats) = num_feats;
    for i = 1:num_feats
        % Note: using add-1 smoothing
        df(1,i) = nnz(train2.counts(:,i)) + 1;
    end
    idf = log(d ./ df);
    [sorted_idf, IDX] = sort(idf);
    idftrain = train2.counts(:,IDX);
    X = idftrain(train_range_start:train_range_end, 1:56835);
    if disp_flag
        disp('    finished.');
    end    
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





%% SVDs(PCA)

if SVDs_flag
    if disp_flag
        disp('Start SVDs(fsvd):');
    end
    for components = 1000  %components used
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
    tic;
    % kernel_libsvm modify to specify C. If you need to use cross
    % validation to determine C, modify back.
    [results.gaussian info.gaussian] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_gaussian, 100);% ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
    toc;
    RMSE_SVM = sqrt(norm(info.gaussian.yhat - Ytest, 2)^2 /length(Ytest));
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
