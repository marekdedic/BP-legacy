"Separates a given URL into 3 parts - domain, query, and path."
function separateUrl(url::AbstractString)::Tuple{AbstractVector{AbstractString}, AbstractVector{AbstractString}, AbstractVector{AbstractString}}
	if contains(url, "://")
		splitted = split(url, "://");
		url = splitted[2];
	end
	splitted = split(url, "/");
	rawDomain = splitted[1];
	# Decode ascii-hex encoded IPs
	if(startswith(rawDomain, "HEX"))
		rawDomain = rawDomain[5:end]
		portSplit = split(rawDomain, ":");
		rawDomain = portSplit[1];
		portSplit = portSplit[2:end];
		IP = ""
		for i::Int in 1:(length(rawDomain)/2)
			c = Char(parse(Int32, rawDomain[(2i - 1):2i], 16));
			IP *= string(c);
		end
		for i in portSplit
			IP *= i;
		end
		domain = Vector{String}();
		push!(domain, IP);
	else
		domain = split(rawDomain, ".");
	end
	splitted = splitted[2:end];
	path = Vector{String}();
	query = Vector{String}();
	if length(splitted) != 0
		splitted2 = split(splitted[end], "?")
		splitted[end] = splitted2[1];
		if length(splitted2) > 1
			query = split(splitted2[2], "&");
		end
		path = splitted;
	end
	# Optional: add empty string when some part is empty array
	if(length(domain) == 0)
		push!(domain, "");
	end
	if(length(path) == 0)
		push!(path, "");
	end
	if(length(query) == 0)
		push!(query, "");
	end
	return (domain, path, query);
end

function processDataset(urls::Vector, labels::Vector; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)::UrlDataset
	featureMatrix = Vector{Vector{T}}(0);
	results = Vector{Int}(0);
	bags = Vector{Int}(0);
	urlParts = Vector{Int}(0);
	info = Vector{AbstractString}(0);
	maxBag = 1;

	#Threads.@threads for j in 1:size(labels, 1)
	for j in 1:size(labels, 1)
		(domain, path, query) = separateUrl(urls[j]);
		for i in domain
			push!(featureMatrix, featureGenerator(i, featureCount; T = T));
			push!(results, labels[j]);
			push!(bags, maxBag);
			push!(urlParts, 1);
			push!(info, urls[j]);
		end
		for i in path
			push!(featureMatrix, featureGenerator(i, featureCount; T = T));
			push!(results, labels[j]);
			push!(bags, maxBag);
			push!(urlParts, 2);
			push!(info, urls[j]);
		end
		for i in query
			push!(featureMatrix, featureGenerator(i, featureCount; T = T));
			push!(results, labels[j]);
			push!(bags, maxBag);
			push!(urlParts, 3);
			push!(info, urls[j]);
		end
		maxBag += 1;
	end
	return UrlDataset(hcat(featureMatrix...),results, bags, urlParts; info = info);
end
