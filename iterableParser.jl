import Base.start, Base.next, Base.done;

type IterableParser
	urls::Vector{AbstractString};
	labels::Vector{Int};
	batchSize::Int;

	featureCount::Int;
	featureGenerator::Function;
	T::DataType;
end

function IterableParser(urls::Vector{AbstractString}, labels::Vector{Int}, batchSize::Int; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)
	return IterableParser(urls, labels, batchSize, featureCount, featureGenerator, T);
end

function start(::IterableParser)::Int
	return 1;
end

function next(iter::IterableParser, state::Int)::Tuple{UrlDataset, Int}
	if iter.batchSize == 0
		start = 1;
		stop = length(iter.labels);
	else
		start = (state - 1) * iter.batchSize + 1;
		stop = min(state * iter.batchSize, length(iter.labels));
	end
	dataset = loadSampleUrl(iter.urls[start:stop], iter.labels[start:stop]; featureCount = iter.featureCount, featureGenerator = iter.featureGenerator, T = iter.T);
	return (dataset, state + 1)
end

function done(iter::IterableParser, state::Int)::Bool
	if iter.batchSize == 0
		return state > 1;
	end
	return state > cld(length(iter.labels), batchSize);
end

function sample(iter::IterableParser)
	perm = StatsBase.sample(1:length(iter.urls), iter.batchSize);
	return loadSampleUrl(iter.urls[perm], iter.labels[perm]; featureCount = iter.featureCount, featureGenerator = iter.featureGenerator, T = iter.T)
end
