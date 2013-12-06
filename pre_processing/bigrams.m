% NOTE: Alter this input to process modified review data.
M = load('../data/metadata.mat');

% Map bit vectors to pairs
data = M.train_metadata;
map = containers.Map('KeyType','char','ValueType','int32');
for i = 1:length(data)
	review = data(i).text;
	fst = 1;
	snd = 2;
	fprintf('%i\n',i);
	while snd < length(review)
		gram = char(strcat(lower(review(fst)),'_',lower(review(snd))));
		if isKey(map,gram)
			vect = map(gram);
			vect(1,i) = 1;
			map(gram) = vect;
		else
			map(gram) = zeros(1,length(data));
		end
		fst = fst + 1;
		snd = snd + 1;
	end	
end

% Construct final matrix
valueset = values(map);
mat = zeros(length(valueset),length(data));
for i = 1:length(valueset)
	mat(:,i) = valueset(i)';
end

save '../data/bigram.mat' mat
