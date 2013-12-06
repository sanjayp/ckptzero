% NOTE: Alter this input to process modified review data.
M = load('../data/metadata.mat');

% Select the total number of samples to use. For the training data,
% the range is 1 to 25000.
SLICE_SIZE = length(M.train_metadata);

% Construct linked lists of data to be made sparse
data = M.train_metadata;
clear M
rowvect = java.util.LinkedList;
colvect = java.util.LinkedList;
counts = java.util.LinkedList;
current_count = 0;
word_map = containers.Map();
for i = 1:SLICE_SIZE
	review = data(i).text;
	fst = 1;
	snd = 2;
	fprintf('%i\n',i);
	submap = containers.Map();
	while snd < length(review)
		gram = char(strcat(lower(review(fst)),'_',lower(review(snd))));
		word_idx = 0;
		if not(isKey(word_map,gram))
			current_count = current_count + 1;
			word_map(gram) = current_count;
		end
		if isKey(submap,gram)
			c = submap(gram);
			submap(gram) = c + 1;	
		else
			submap(gram) = 1;
		end
		fst = fst + 1;
		snd = snd + 1;
	end	
	keyset = keys(submap);
	for j = 1:length(keyset)
		gram = char(keyset(j));
		idx = word_map(gram);
		rowvect.add(i);
		colvect.add(idx);
		counts.add(submap(gram));
	end
end

% Clear unnecessary data
clear data word_map

% Construct final matrix
rowvect = cell2mat(cell(rowvect.toArray(rowvect)));
colvect = cell2mat(cell(colvect.toArray(colvect)));
counts = cell2mat(cell(counts.toArray(counts)));
bigram_data = sparse(rowvect, colvect, counts, SLICE_SIZE, current_count);

save '../data/bigram.mat' bigram_data
