using EduNets;

import Base.Operators.getindex, Base.vcat
import EduNets.sample
import StatsBase.sample

type UrlDataset{T<:AbstractFloat}<:AbstractDataset
	domains::SortedSingleBagDataset{T}
	paths::SortedSingleBagDataset{T}
	queries::SortedSingleBagDataset{T}

	y::AbstractVector{Int}
	info::DataFrames.DataFrame;
end

function UrlDataset(features::Matrix, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; info::Vector{AbstractString} = Vector{AbstractString}(0), T::DataType = Float32)::UrlDataset
	if(!issorted(urlIDs))
		permutation = sortperm(urlIDs);
		features = features[:, permutation];
		labels = labels[permutation];
		urlIDs = urlIDs[permutation];
		urlParts = urlParts[permutation];
		if size(info, 1) != 0;
			info = info[permutation];
		end
	end

	(domainFeatures, pathFeatures, queryFeatures) = map(i->features[:, urlParts .== i], 1:3);
	(domainBags, pathBags, queryBags) = map(i->findranges(urlIDs[urlParts .== i]), 1:3);

	subbags = findranges(urlIDs);
	bagLabels = map(b->maximum(labels[b]), subbags);
	if size(info, 1) != 0;
		bagInfo = map(b->info[b][1], subbags);
	else
		bagInfo = Vector{AbstractString}(0);
	end

	domains = SortedSingleBagDataset(domainFeatures, bagLabels, domainBags);
	paths = SortedSingleBagDataset(pathFeatures, bagLabels, pathBags);
	queries = SortedSingleBagDataset(queryFeatures, bagLabels, queryBags);
	UrlDataset(domains, paths, queries, bagLabels, DataFrames.DataFrame(url = bagInfo))
end

function featureSize(dataset::UrlDataset)::Int
	size(dataset.domains.x, 1)
end

function getindex(dataset::UrlDataset, i::Int)
	getindex(dataset, [i])
end

function getindex(dataset::UrlDataset, indices::AbstractArray{Int})
	if size(dataset.info, 1) == 0
		info = DataFrames.DataFrame([]);
	else
		info = dataset.info[indices, :];
	end
	UrlDataset(dataset.domains[indices], dataset.paths[indices], dataset.queries[indices], dataset.y[indices], info)
end

function vcat(d1::UrlDataset,d2::UrlDataset)
	UrlDataset(vcat(d1.domains,d2.domains), vcat(d1.paths,d2.paths), vcat(d1.queries,d2.queries), vcat(d1.y,d2.y), vcat(d1.info, d2.info))
end

function sample(ds::UrlDataset,n::Int64)
  indexes=sample(1:length(ds.y),min(n,length(ds.y)),replace=false);
  return(getindex(ds,indexes));
end

function sample(ds::UrlDataset,n::Array{Int64})
  classbagids=map(i->findn(ds.y.==i),1:maximum(ds.y));
  indexes=mapreduce(i->sample(classbagids[i],minimum([length(classbagids[i]),n[i]]);replace=false),append!,1:min(length(classbagids),length(n)));
  return(getindex(ds,indexes));
end

