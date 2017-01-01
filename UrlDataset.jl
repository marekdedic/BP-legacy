push!(LOAD_PATH, "EduNets/src");

import GZip
import JSON
using  EduNets

type UrlDataset{T<:AbstractFloat}<:AbstractDataset
	"The feature matrix"
	features::AbstractMatrix{T};
	"Label vector whose size is the number of bags - each element of y corresponds to an element of bags"
	labels::AbstractVector{T};

	"Each element of bags is one bag. Its elements are indices of lines from features, which are in this bag."
	bags::Vector{Vector{Int}};
	"Labeling of URL parts: 1 - domain; 2 - path; 3 - query"
	urlParts::Vector{Int};
	#bags::Vector{Vector{Tuple{Int, Vector{Int}}}}; # unified alternative

	#info::DataFrames.DataFrame; # no idea what this is for
end;

function UrlDataset(features::Matrix, labels::Vector, bagIDs, urlParts::Vector{Int})::UrlDataset
	bagMap = Dict{eltype(bagIDs), Int}();
	for i in 1:length(bagIDs)
		if(!haskey(bagMap, bagIDs[i]))
			bagMap[bagIDs[i]] = [i];
		else
			push!(bagMap[BagIDs[i]], i)
		end
	end
	bags = Vector{Vector{Int}}(length(bagMap));
	bagLabels = Vector{Int}(length(bagMap))
	for i in 1:length(bagMap)
		bags[i] = bagMap[i][2];
		bagLabels[i] = maximum(labels[bagMap[i][2]]);
	end
	return UrlDataset(features, bagLabels, bags, urlParts);
end


# LEGACY CODE:

type Url
	protocol::AbstractString
	domain::AbstractArray{AbstractString, 1};
	port::Int64;
	path::AbstractArray{AbstractString, 1};
	query::AbstractArray{Tuple{AbstractString, AbstractString}, 1};
end;
Url() = Url("", [], 80, [], []);

function readURL(input::AbstractString)::Url
  output::Url = Url();
  splitted = split(input, "://");
  output.protocol = splitted[1];
  if in(':', splitted[2])
	  splitted = split(splitted[2], ":");
	  domain = splitted[1];
	  splitted = split(splitted[2], "/");
	  output.port = parse(Int64, splitted[1]);
	  splice!(splitted, 1);
  end
  query = split(splitted[end], "?");
  splitted[end] = query[1];
  query = query[2];
  for s in split(domain, ".")
	  push!(output.domain, s);
  end
  for s in splitted
	  push!(output.path, s);
  end
  for s in split(query, "&")
	  keyvalue = split(s, "=");
	  push!(output.query, (keyvalue[1], keyvalue[2]));
  end
  println(output);
  return output;
end

