import GZip;
import JSON;
import JLD;
import DataFrames;
using EduNets;

include("UrlDataset.jl");

# Helper functions

"Generates an array of all the n-grams (substrings of length n) from a given string."
function ngrams(input::AbstractString, n::Int)::Vector{AbstractString}
	output = Vector{AbstractString}(0);
	for (i, c) in enumerate(input)
		push!(output, AbstractString(""));
		output[i] = string(output[i], c);
		for j in 1:(n - 1)
			if i > j
				output[i - j] = string(output[i - j], c);
			end
		end
	end
	return output;
end

trigrams(input::AbstractString)::Array{AbstractString} = ngrams(input, 3);

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

"Loads all URLs from a given JSON (.joy.json.gz) file."
function loadUrlFromJSON(file::AbstractString)::Vector{AbstractString}
	output = Vector{AbstractString}(0);
	GZip.open(file) do g
		for line in eachline(g)
			json = JSON.parse(line);
			try;
				ohttp = json["ohttp"];
				if ohttp != nothing
					url = ohttp["Host"] * ohttp["uri"];
					push!(output, url);
				end
			end
		end
	end
	return output;
end

"Sets up a global dictionary for AVClassResultParser, so that it doesn't have to be loaded every time."
function setupAVClass(file::AbstractString)
	global avclass_dict = Dict{String, String}();
	open(file) do file
		for line in eachline(file)
			values = split(chomp(line), '\t');
			if size(values)[1] > 2
				avclass_dict[values[1]] = values[3];
			end
		end
	end
end

# Feature generation functions

function trigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32)::Array{Float32}
	output = zeros(T, modulo);
	for i in trigrams(input)
		index = mod(hash(i), modulo);
		output[index + 1] += 1;
	end
	return output;
end

# Labeling functions

function countingResultParser(file::AbstractString; threshold::Int = 5)::Int
	positiveCount = 0;
	open(file) do f
		json = JSON.parse(f);
		try
			scans = json["scans"];
			try
				if(scans["Malwarebytes"]["detected"] == true)
					positiveCount += 1;
				end
			end
			try
				if(scans["BitDefender"]["detected"] == true)
					positiveCount += 1;
				end
			end  
			try
				if(scans["Symantec"]["detected"] == true)
					positiveCount += 1;
				end
			end
			try
				if(scans["ESET-NOD32"]["detected"] == true)
					positiveCount += 1;
				end
			end
			try
				if(scans["Avast"]["detected"] == true)
					positiveCount += 1;
				end
			end
			try
				if(scans["Kaspersky"]["detected"] == true)
					positiveCount += 1;
				end
			end
			try
				if(scans["Avira"]["detected"] == true)
					positiveCount += 1;
				end
			end
			try
				if(scans["AVG"]["detected"] == true)
					positiveCount += 1;
				end
			end
		end
	end
	return positiveCount >= threshold ? 2 : 1;
end

function AVClassResultParser(file::AbstractString)::Int
	result = get(avclass_dict, file, "");
	return (result == "CLEAN" || result == "") ? 1 : 2;
end

# ThreatgridSample loading functions

function loadThreatGrid(dir::AbstractString; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, resultParser::Function = countingResultParser, T::DataType = Float32)::SingleBagDataset
	featureMatrix = [Vector{Vector{T}}(0) for i in 1:Threads.nthreads()];
	results = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	bags = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	maxBag = [1 for i in 1:Threads.nthreads()];
	aggregatedFeatures = Matrix{T}(featureCount, 0);
	aggregatedResults = Vector{Int}(0);
	aggregatedBags = Vector{Int}(0);
	for (root, dirs, files) in walkdir(dir)
		Threads.@threads for file in filter(x-> ismatch(r"\.joy\.json\.gz$", x), files)
		path = joinpath(root, file);
			filename = replace(path, r"^(.*)\.joy\.json\.gz$", s"\1");
			if isfile(filename * ".vt.json")
				urls = loadUrlFromJSON(filename * ".joy.json.gz");
				result = resultParser(filename * ".vt.json");
				for url in urls
					push!(featureMatrix[Threads.threadid()], featureGenerator(url, featureCount; T = T));
					push!(results[Threads.threadid()], result);
					push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
				end
			end
			maxBag[Threads.threadid()] += 1;
		end
	end
	for i in 1:Threads.nthreads()
		features = hcat(featureMatrix[i]...);
		if length(size(features)) != 2
			continue;
		end
		aggregatedFeatures = hcat(aggregatedFeatures, features);
		aggregatedResults = vcat(aggregatedResults, results[i]);
		bags[i] += size(aggregatedBags)[1];
		aggregatedBags = vcat(aggregatedBags, bags[i]);
	end
	return SingleBagDataset(aggregatedFeatures, aggregatedResults, aggregatedBags);
end

function loadThreatGridUrl(dir::AbstractString; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, resultParser::Function = countingResultParser, T::DataType = Float32)::UrlDataset
	featureMatrix = [Vector{Vector{T}}(0) for i in 1:Threads.nthreads()];
	results = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	bags = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	urlParts = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	maxBag = [1 for i in 1:Threads.nthreads()];
	aggregatedFeatures = Matrix{T}(featureCount, 0);
	aggregatedResults = Vector{Int}(0);
	aggregatedBags = Vector{Int}(0);
	aggregatedUrlParts = Vector{Int}(0);
	for (root, dirs, files) in walkdir(dir)
		Threads.@threads for file in filter(x-> ismatch(r"\.joy\.json\.gz$", x), files)
			path = joinpath(root, file);
			filename = replace(path, r"^(.*)\.joy\.json\.gz$", s"\1");
			if isfile(filename * ".vt.json")
				urls = loadUrlFromJSON(filename * ".joy.json.gz");
				result = resultParser(filename * ".vt.json");
				for url in urls
					(domain, path, query) = separateUrl(url);
					for i in domain
						push!(featureMatrix[Threads.threadid()], featureGenerator(i, featureCount; T = T));
						push!(results[Threads.threadid()], result);
						push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
						push!(urlParts[Threads.threadid()], 1);
					end
					for i in path
						push!(featureMatrix[Threads.threadid()], featureGenerator(i, featureCount; T = T));
						push!(results[Threads.threadid()], result);
						push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
						push!(urlParts[Threads.threadid()], 2);
					end
					for i in query
						push!(featureMatrix[Threads.threadid()], featureGenerator(i, featureCount; T = T));
						push!(results[Threads.threadid()], result);
						push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
						push!(urlParts[Threads.threadid()], 3);
					end
					maxBag[Threads.threadid()] += 1;
				end
			end
		end
	end
	for i in 1:Threads.nthreads()
		features = hcat(featureMatrix[i]...);
		if length(size(features)) != 2
			continue;
		end
		aggregatedFeatures = hcat(aggregatedFeatures, features);
		aggregatedResults = vcat(aggregatedResults, results[i]);
		bags[i] += size(aggregatedBags)[1];
		aggregatedBags = vcat(aggregatedBags, bags[i]);
		aggregatedUrlParts = vcat(aggregatedUrlParts, urlParts[i]);
	end
	return UrlDataset(aggregatedFeatures, aggregatedResults, aggregatedBags, aggregatedUrlParts);
end

# Sample loading function
function loadSampleUrlThreaded(file::AbstractString; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, resultParser::Function = countingResultParser, T::DataType = Float32)::UrlDataset
	featureMatrix = [Vector{Vector{T}}(0) for i in 1:Threads.nthreads()];
	results = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	bags = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	urlParts = [Vector{Int}(0) for i in 1:Threads.nthreads()];
	info = [Vector{AbstractString}(0) for i in 1:Threads.nthreads()];
	maxBag = [1 for i in 1:Threads.nthreads()];
	aggregatedFeatures = Matrix{T}(featureCount, 0);
	aggregatedResults = Vector{Int}(0);
	aggregatedBags = Vector{Int}(0);
	aggregatedUrlParts = Vector{Int}(0);
	aggregatedInfo = Vector{AbstractString}(0);

	table = GZip.open(file,"r") do fid
		readcsv(fid)
	end
	if any(table[:, 3].!="legit")
		table=table[table[:, 3].!="legit",:]
	end

	table=table[1:min(size(table,1),6000),:]
	urls = table[:, 1];
	labels = (table[:, 3].!="legit")+1;
	#Threads.@threads for j in 1:size(labels, 1)
	for j in 1:size(labels, 1)
		(domain, path, query) = separateUrl(urls[j]);
		for i in domain
			push!(featureMatrix[Threads.threadid()], featureGenerator(i, featureCount; T = T));
			push!(results[Threads.threadid()], labels[j]);
			push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
			push!(urlParts[Threads.threadid()], 1);
			push!(info[Threads.threadid()], urls[j]);
		end
		for i in path
			push!(featureMatrix[Threads.threadid()], featureGenerator(i, featureCount; T = T));
			push!(results[Threads.threadid()], labels[j]);
			push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
			push!(urlParts[Threads.threadid()], 2);
			push!(info[Threads.threadid()], urls[j]);
		end
		for i in query
			push!(featureMatrix[Threads.threadid()], featureGenerator(i, featureCount; T = T));
			push!(results[Threads.threadid()], labels[j]);
			push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
			push!(urlParts[Threads.threadid()], 3);
			push!(info[Threads.threadid()], urls[j]);
		end
		maxBag[Threads.threadid()] += 1;
	end
	for i in 1:Threads.nthreads()
		features = hcat(featureMatrix[i]...);
		if length(size(features)) != 2
			continue;
		end
		aggregatedFeatures = hcat(aggregatedFeatures, features);
		aggregatedResults = vcat(aggregatedResults, results[i]);
		bags[i] += size(aggregatedBags)[1];
		aggregatedBags = vcat(aggregatedBags, bags[i]);
		aggregatedUrlParts = vcat(aggregatedUrlParts, urlParts[i]);
		aggregatedInfo = vcat(aggregatedInfo, info[i]);
	end
	return UrlDataset(aggregatedFeatures, aggregatedResults, aggregatedBags, aggregatedUrlParts; info = aggregatedInfo);
end

type SampleBatchParser
	urls::Vector{AbstractString};
	labels::Vector{Int};
	batchSize::Int;
	currentBatch::Int;

	featureCount::Int;
	featureGenerator::Function;
	T::DataType;
end

function SampleBatchParser(urls::Vector{AbstractString}, labels::Vector{Int}, batchSize::Int; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)
	return SampleBatchParser(urls, labels, batchSize, 1, featureCount, featureGenerator, T);
end

# Sample loading function - non-threaded version
function loadSampleUrl(file::AbstractString; batchSize::Int = 6000, featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)::SampleBatchParser
	table = GZip.open(file,"r") do fid
		readcsv(fid)
	end
	if any(table[:, 3].!="legit")
		table=table[table[:, 3].!="legit",:]
	end

	urls = table[:, 1];
	labels = (table[:, 3].!="legit")+1;
	return SampleBatchParser(convert(Vector{AbstractString}, urls), convert(Vector{Int}, labels), batchSize; featureCount = featureCount, featureGenerator = featureGenerator, T = T)
end

function nextBatch!(sbp::SampleBatchParser)::Tuple{Bool, UrlDataset}
	if sbp.batchSize == 0
		return (true, loadSampleUrl(sbp.urls, sbp.labels; featureCount = sbp.featureCount, featureGenerator = sbp.featureGenerator, T = sbp.T));
	else
		start = min((sbp.currentBatch - 1) * sbp.batchSize + 1, length(sbp.urls));
		stop = min(sbp.currentBatch * sbp.batchSize, length(sbp.urls));
		if start == stop
			return (false, loadSampleUrl(sbp.urls[start:stop], sbp.labels[start:stop]; featureCount = sbp.featureCount, featureGenerator = sbp.featureGenerator, T = sbp.T));
		else
			sbp.currentBatch += 1;
			return (true, loadSampleUrl(sbp.urls[start:stop], sbp.labels[start:stop]; featureCount = sbp.featureCount, featureGenerator = sbp.featureGenerator, T = sbp.T));
		end
	end
end

function loadSampleUrl(urls::Vector, labels::Vector; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)::UrlDataset
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

# Actual realisations of a complete dataset parser.

function parseDataset(dir::AbstractString, file::AbstractString = "dataset.jld")::Void
	JLD.save(file, "dataset", loadThreatGrid(dir));
	return nothing;
end

function parseDatasetAVClass(dir::AbstractString, file::AbstractString = "dataset.jld")::Void
	setupAVClass("avclass_results.txt");
	dataset = loadThreatGrid(dir, resultParser = AVClassResultParser);
	JLD.save(file, "dataset", dataset);
	return nothing;
end

function parseDatasetAVClassUrl(dir::AbstractString, file::AbstractString = "dataset.jld")::Void
	setupAVClass("avclass_results.txt");
	dataset = loadThreatGridUrl(dir, resultParser = AVClassResultParser);
	JLD.save(file, "dataset", dataset);
	return nothing;
end

function parseSampleUrl(source::AbstractString, file::AbstractString = "dataset.jld")::Void
	sbp = loadSampleUrl(source; batchSize = 0);
	dataset = nextBatch!(sbp);
	JLD.save(file, "dataset", dataset);
	return nothing;
end

