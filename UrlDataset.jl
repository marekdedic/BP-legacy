push!(LOAD_PATH, "EduNets/src");

import GZip
import JSON
using  EduNets

type UrlDataset{T<:AbstractFloat}<:AbstractDataset
	"Features extracted from domain part of url"
	domainFeatures::AbstractMatrix{T};
	"Features extracted from path part of url"
	pathFeatures::AbstractMatrix{T};
	"Features extracted from query part of url"
	queryFeatures::AbstractMatrix{T};
	"Label vector whose size is the number of bags - each element of y corresponds to an element of bags"
	labels::AbstractVector{Int};

	"Each element of bags is one bag. Its elements are indices of lines from features, which are in this bag."
	bags::Vector{Vector{Int}};

	#info::DataFrames.DataFrame; # Additional metadata, not used
end;

	"Labeling of URL parts: 1 - domain; 2 - path; 3 - query"
function UrlDataset(features::Matrix, labels::Vector, bagIDs, urlParts::Vector{Int})::UrlDataset
	bagMap = Dict{eltype(bagIDs), Vector{Int}}();
	for i in 1:length(bagIDs)
		if(!haskey(bagMap, bagIDs[i]))
			bagMap[bagIDs[i]] = [i];
		else
			push!(bagMap[bagIDs[i]], i)
		end
	end
	bags = Vector{Vector{Int}}(length(bagMap));
	bagLabels = Vector{Int}(length(bagMap))
	for (i, it) in enumerate(bagMap)
		bags[i] = it[2];
		bagLabels[i] = maximum(labels[it[2]]);
	end
	domainFeatures = Matrix{AbstractFloat}(size(features)[1], 0);
	pathFeatures = Matrix{AbstractFloat}(size(features)[1], 0);
	queryFeatures = Matrix{AbstractFloat}(size(features)[1], 0);
	for i in 1:length(urlParts)
		if(urlParts[i] == 1)
			domainFeatures = hcat(domainFeatures, features[:,i]);
		elseif(urlParts[i] == 2)
			pathFeatures = hcat(pathFeatures, features[:,i]);
		elseif(urlParts[i] == 3)
			queryFeatures = hcat(queryFeatures, features[:,i]);
		end
	end
	return UrlDataset(domainFeatures, pathFeatures, queryFeatures, bagLabels, bags);
end

function separateUrl(url::AbstractString)::Tuple{String, String, String}
	protocol = "http";
	if contains(url, "://")
		splitted = split(url, "://");
		protocol = splitted[1];
		url = splitted[2];
	end
	splitted = split(url, "/");
	domain = splitted[1];
	splitted = splitted[2:end];
	path = "";
	query = "";
	if length(splitted) != 0
		splitted2 = split(splitted[end], "?")
		splitted[end] = splitted2[1];
		if length(splitted2) > 1
			query = splitted2[2]
		end
		path = join(splitted, "/")
	end
	return (protocol * "://" * domain, path, query);
end

