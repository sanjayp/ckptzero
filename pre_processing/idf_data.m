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


