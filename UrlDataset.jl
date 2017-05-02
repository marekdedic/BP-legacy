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
	permutation = sortperm(urlIDs);
	features = features[:, permutation];
	labels = labels[permutation];
	urlIDs = urlIDs[permutation];
	urlParts = urlParts[permutation];
	if size(info, 1) != 0;
		info = info[permutation];
	end
	subbags = findranges(urlIDs);

	domainFeatures = Vector{Vector{T}}(0);
	pathFeatures = Vector{Vector{T}}(0);
	queryFeatures = Vector{Vector{T}}(0);
	bagLabels = Vector{Int}(length(subbags));
	if size(info, 1) != 0;
		bagInfo = Vector{AbstractString}(length(subbags));
	else
		bagInfo = Vector{AbstractString}(0);
	end
	# TODO: Implement bags
	bags = Vector{UnitRange{Int}}(length(subbags));

	for (i, r) in enumerate(subbags)
		for (j, part) in enumerate(urlParts[r])
			if part == 1
				push!(domainFeatures, features[:, first(r) + j - 1]);
			elseif part == 2
				push!(pathFeatures, features[:, first(r) + j - 1]);
			elseif part == 3
				push!(queryFeatures, features[:, first(r) + j - 1]);
			end
		end
		bagLabels[i] = maximum(labels[r]);
		if size(info, 1) != 0;
			bagInfo[i] = info[r][1];
		end
		bags[i] = i:i;
	end

	domains = SortedSingleBagDataset(hcat(domainFeatures...), bagLabels, bags);
	paths = SortedSingleBagDataset(hcat(pathFeatures...), bagLabels, bags);
	queries = SortedSingleBagDataset(hcat(queryFeatures...), bagLabels, bags);
	UrlDataset(domains, paths, queries, bagLabels, convert(DataFrames.DataFrame, reshape(bagInfo, length(bagInfo), 1)))
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

