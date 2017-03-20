using EduNets;

import Base.Operators.getindex, Base.vcat
import EduNets.sample
import StatsBase.sample

type UrlDataset{T<:AbstractFloat}<:AbstractDataset
	domains::SortedSingleBagDataset{T}
	paths::SortedSingleBagDataset{T}
	queries::SortedSingleBagDataset{T}

	y::AbstractVector{Int}
end

# TODO: Handle empty bags

function UrlDataset(features::Matrix, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; T::DataType = Float32)::UrlDataset
	permutation = sortperm(urlIDs);
	features = features[:, permutation];
	labels = labels[permutation];
	urlIDs = urlIDs[permutation]
	urlParts = urlParts[permutation];
	subbags = findranges(urlIDs);

	domainFeatures = Vector{Vector{T}}(0);
	pathFeatures = Vector{Vector{T}}(0);
	queryFeatures = Vector{Vector{T}}(0);
	bagLabels = Vector{Int}(length(subbags));
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
		bags[i] = i:i;
	end

	domains = SortedSingleBagDataset(hcat(domainFeatures...), bagLabels, bags);
	paths = SortedSingleBagDataset(hcat(pathFeatures...), bagLabels, bags);
	queries = SortedSingleBagDataset(hcat(queryFeatures...), bagLabels, bags);
	UrlDataset(domains, paths, queries, bagLabels)
end

function featureSize(dataset::UrlDataset)::Int
	size(dataset.domains.x, 1)
end

function getindex(dataset::UrlDataset, i::Int)
	getindex(dataset, [i])
end

function getindex(dataset::UrlDataset, indices::AbstractArray{Int})
	UrlDataset(dataset.domains[indices], dataset.paths[indices], dataset.queries[indices], dataset.y[indices])
end

function vcat(d1::UrlDataset,d2::UrlDataset)
	UrlDataset(vcat(d1.domains,d2.domains), vcat(d1.paths,d2.paths), vcat(d1.queries,d2.queries), vcat(d1.y,d2.y))
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

