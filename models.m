


%%

clear all;
%% Flag
freq_flag = 1;
stemmer_flag = 0;
idf_flag = 1;
SVDs_flag = 0;
SVM_flag = 0;
porterStemmer_flag = 0;
LG_flag = 1;


%% Load Data

load ../data/review_dataset.mat;


% reset train data name to train2 to avoid confliction with func train
train2 = train;
clearvars train;

%% set ind

train_range_start = 1;
train_range_end = 24000;
test_range_start = 24001;
test_range_end = 25000;

%% Default 25000 * 65000

    if 1
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
        

%% use word frequency
if freq_flag
    total = cat(1, train2.counts, quiz.counts);
    freq = sum(total);
    RMSE_ind = 0;

    for f = 10
        RMSE_ind = RMSE_ind + 1;

        high_freq = find(freq>=f);  %find high frequency count
        low_freq = find(freq<f);    %find low frequency count
        highfreq_count = train2.counts(:,high_freq);
        lowfreq_count = train2.counts(:, low_freq);
        lowfreq_sum = sum(lowfreq_count')'; %sum up the low frequency as a feature

        X = highfreq_count(1:24000,:);      %train data 
        Y = train2.labels(1:24000,:);       %train labels
        Xtest = highfreq_count(end - 1000:end,:);   %test data
        Ytest = train2.labels(end - 1000:end,:);    %test label

    end
end

%% Construct idf vector for features
if idf_flag
    num_feats = length(train2.counts);
    df = zeros(1,num_feats);
    d = zeros(1,num_feats);
    d(1:num_feats) = num_feats;
    for i = 1:num_feats
        %% Note: using add-1 smoothing
        df(1,i) = nnz(train2.counts(:,i)) + 1;
    end
    idf = log(d ./ df);
    [sorted_idf, IDX] = sort(idf);
    idftrain = train2.counts(:,IDX);
    X = idftrain(train_range_start:train_range_end, 1:56835);
    
end


%% porter stemmer

if porterStemmer_flag
    stem_map = zeros(1, length(vocab));
    stem_i = 0;

    for iterator = 1:1:length(vocab)
        if strcmp(vocab{iterator}, 'aed')
            continue;
        end
        % if the word need to be stemmed
        if ~strcmp(porterStemmer(vocab{iterator}), vocab{iterator})
            %
            stem_i = stem_i +1;
            stemmed_to_ind = find(ismember(vocab, porterStemmer(vocab{iterator}) ));
            if isempty(stemmed_to_ind)
                stem_map(iterator) = iterator;
            else
                %disp(stemmed_to_ind);
                %disp(vocab{iterator});
                stem_map(iterator) = stemmed_to_ind;
                disp(stemmed_to_ind);
                disp(iterator);
                X(:, stemmed_to_ind) = X(:, stemmed_to_ind) + X(:, iterator);
                % delete or just set to zero?
                X(:, iterator) = 0;
                %disp(vocab{stemmed_to_ind});
            end
        else
            stem_map(iterator) = iterator;
        end
    end
 
end    %end of the flag


%% use porterStemmer



%% SVDs(PCA)

if SVDs_flag
    for components = 10:10:100  %components used
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
        X = X_new;
        Xtest = X_test_new;
    end
end
%% SVM

if SVM_flag
    clearvars kernel_gaussian;
    kernel_gaussian = @(x,x2) kernel_gaussian(x, x2, 20);
    tic;
    [results.gaussian info.gaussian] = kernel_libsvm(Xnew, Y, XTestNew, Ytest, kernel_gaussian, 100);% ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
    toc;
    RMSE333 = sqrt(norm(info.gaussian.yhat - Ytest, 2)^2 /length(Ytest));
    disp(RMSE333);
end

%% Logistic Regression
if LG_flag
    model = train(Y, X, '-s 0 -q');
    [prediction, accuracy, dec_values] = predict(Ytest, Xtest, model); % test the training data
    RMSE892 = sqrt(norm(prediction - Ytest, 2)^2 /length(Ytest));
    disp(RMSE892);
end
