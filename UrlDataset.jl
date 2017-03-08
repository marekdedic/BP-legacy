using EduNets;

import Base.Operators.getindex

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

