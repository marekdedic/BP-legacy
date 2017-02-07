push!(LOAD_PATH, "EduNets/src");

type UrlDatasetCompound{T<:AbstractFloat}<:AbstractDataset
	domains::SortedSingleBagDataset{T}
	paths::SortedSingleBagDataset{T}
	queries::SortedSingleBagDataset{T}

	labels::AbstractVector{Int}
end

# TODO: Handle empty bags

function UrlDatasetCompound(features::Matrix, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; T::DataType = Float64)::UrlDatasetCompound
	permutation = sortperm(urlIDs);
	fetures = features[:, permutation];
	labels = labels[permutation];
	urlIDs = urlIDs[permutation]
	urlParts = urlParts[permutation];
	subbags = findranges(urlIDs);

	domainFeatures = Matrix{T}(size(features)[1], 0);
	pathFeatures = Matrix{T}(size(features)[1], 0);
	queryFeatures = Matrix{T}(size(features)[1], 0);
	bagLabels = Vector{Int}(length(subbags));
	# TODO: Implement bags
	bags = Vector{UnitRange{Int}}(length(subbags));

	for (i, r) in enumerate(subbags)
		if length(r) != 3
			error("Invalid data - found an url which doesn't have 3 parts!");
		end
		for (j, part) in enumerate(urlParts[r])
			if part == 1
				domainFeatures = hcat(domainFeatures, features[:, first(r) + j - 1]);
			elseif part == 2
				pathFeatures = hcat(pathFeatures, features[:, first(r) + j - 1]);
			elseif part == 1
				queryFeatures = hcat(queryFeatures, features[:, first(r) + j - 1]);
			end
		end
		bagLabels[i] = maximum(labels[r]);
		bags[i] = i:i;
	end

	domains = SortedSingleBagDataset(domainFeatures, bagLabels, bags);
	paths = SortedSingleBagDataset(domainFeatures, bagLabels, bags);
	queries = SortedSingleBagDataset(domainFeatures, bagLabels, bags);
	UrlDatasetCompound(domains, paths, queries, bagLabels)
end
