% NOTE: Alter this input to process modified review data.
M = load('../data/metadata.mat');

% Select the total number of samples to use. For the training data,
% the range is 1 to 25000.
SLICE_SIZE = 100;

% Map bit vectors to pairs
data = M.train_metadata;
clear M
map = containers.Map();
for i = 1:SLICE_SIZE
	review = data(i).text;
	fst = 1;
	snd = 2;
	% fprintf('%i\n',i);
	while snd < length(review)
		gram = char(strcat(lower(review(fst)),'_',lower(review(snd))));
		if isKey(map,gram)
			vect = map(gram);
			vect(1,i) = 1;
			map(gram) = vect;
		else
			map(gram) = zeros(1,SLICE_SIZE);
		end
		fst = fst + 1;
		snd = snd + 1;
	end	
end

% Construct final matrix
clear data
valueset = values(map);
clear map
mat = sparse(vertcat(valueset{:})');

save '../data/bigram.mat' mat
