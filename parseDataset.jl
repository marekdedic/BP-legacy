push!(LOAD_PATH, "EduNets/src");

import GZip
import JSON
import JLD
using EduNets

function ngrams(input::AbstractString, n::Int64)::Array{AbstractString}
	output = Array{AbstractString}(0);
	i = 1;
	for c in input
		push!(output, AbstractString(""));
		output[i] = string(output[i], c);
		for j in 1:(n - 1)
			if i > j
				output[i - j] = string(output[i - j], c);
			end
		end
		i += 1;
	end
	return output;
end

trigrams(input::AbstractString) = ngrams(input, 3);

function trigramFeatureGenerator(input::AbstractString, modulo::Int64)::Array{Float32}
	output = spzeros(Float32, modulo);
	for i in trigrams(input)
		index = mod(hash(i), modulo);
		output[index + 1] += 1;
	end
	return output;
end

function loadUrlFromJSON(file::AbstractString)::Array{AbstractString}
	output = Array{AbstractString}(0);
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

function countingResultParser(file::AbstractString)::Int64
	positiveCount= 0;
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
	return positiveCount >= 5 ? 2 : 1;
end

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

function AVClassResultParser(file::AbstractString)::Int64
	result = get(avclass_dict, file, "");
	return (result == "CLEAN" || result == "") ? 1 : 2;
end

function loadThreatGrid(dir::AbstractString; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, resultParser::Function = countingResultParser)::SingleBagDataset
	featureMatrix = [Array{Float32, 2}(featureCount, 0) for i in 1:Threads.nthreads()];
	results = [Array{Int64}(0) for i in 1:Threads.nthreads()];
	bags = [Array{Int}(0) for i in 1:Threads.nthreads()];
	maxBag = [1 for i in 1:Threads.nthreads()];
	aggregatedFeatures = Array{Float32, 2}(featureCount, 0);
	aggregatedResults = Array{Int64}(0);
	aggregatedBags = Array{Int}(0);
	for (root, dirs, files) in walkdir(dir)
		Threads.@threads for file in filter(x-> ismatch(r"\.joy\.json\.gz$", x), files)
			path = joinpath(root, file);
			filename = replace(path, r"^(.*)\.joy\.json\.gz$", s"\1");
			if isfile(filename * ".vt.json")
				urls = loadUrlFromJSON(filename * ".joy.json.gz");
				result = resultParser(filename * ".vt.json");
				for url in urls
					featureMatrix[Threads.threadid()] = hcat(featureMatrix[Threads.threadid()], featureGenerator(url, featureCount));
					push!(results[Threads.threadid()], result);
					push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
				end
			end
			maxBag[Threads.threadid()] += 1;
		end
	end
	for i in 1:Threads.nthreads()
		aggregatedFeatures = hcat(aggregatedFeatures, featureMatrix[i]);
		aggregatedResults = vcat(aggregatedResults, results[i]);
		bags[i] += size(aggregatedBags)[1];
		aggregatedBags = vcat(aggregatedBags, bags[i]);
	end
	return SingleBagDataset(aggregatedFeatures, aggregatedResults, aggregatedBags);
end

function loadThreatGridUrl(dir::AbstractString; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, resultParser::Function = countingResultParser)::UrlDatasetCompound
	featureMatrix = [Array{Float32, 2}(featureCount, 0) for i in 1:Threads.nthreads()];
	results = [Array{Int64}(0) for i in 1:Threads.nthreads()];
	bags = [Array{Int}(0) for i in 1:Threads.nthreads()];
	urlParts = [Array{Int}(0) for i in 1:Threads.nthreads()];
	maxBag = [1 for i in 1:Threads.nthreads()];
	aggregatedFeatures = Array{Float32, 2}(featureCount, 0);
	aggregatedResults = Array{Int64}(0);
	aggregatedBags = Array{Int}(0);
	aggregatedUrlParts = Array{Int}(0);
	for (root, dirs, files) in walkdir(dir)
		Threads.@threads for file in filter(x-> ismatch(r"\.joy\.json\.gz$", x), files)
			path = joinpath(root, file);
			filename = replace(path, r"^(.*)\.joy\.json\.gz$", s"\1");
			if isfile(filename * ".vt.json")
				urls = loadUrlFromJSON(filename * ".joy.json.gz");
				result = resultParser(filename * ".vt.json");
				for url in urls
					(domain, path, query) = separateUrl(url);
					featureMatrix[Threads.threadid()] = hcat(featureMatrix[Threads.threadid()], featureGenerator(domain, featureCount));
					featureMatrix[Threads.threadid()] = hcat(featureMatrix[Threads.threadid()], featureGenerator(path, featureCount));
					featureMatrix[Threads.threadid()] = hcat(featureMatrix[Threads.threadid()], featureGenerator(query, featureCount));
					push!(results[Threads.threadid()], result);
					push!(results[Threads.threadid()], result);
					push!(results[Threads.threadid()], result);
					push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
					push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
					push!(bags[Threads.threadid()], maxBag[Threads.threadid()]);
					push!(urlParts[Threads.threadid()], 1);
					push!(urlParts[Threads.threadid()], 2);
					push!(urlParts[Threads.threadid()], 3);
				end
			end
			maxBag[Threads.threadid()] += 1;
		end
	end
	for i in 1:Threads.nthreads()
		aggregatedFeatures = hcat(aggregatedFeatures, featureMatrix[i]);
		aggregatedResults = vcat(aggregatedResults, results[i]);
		bags[i] += size(aggregatedBags)[1];
		aggregatedBags = vcat(aggregatedBags, bags[i]);
		aggregatedUrlParts = vcat(aggregatedUrlParts, urlParts[i]);
	end
	return UrlDatasetCompound(aggregatedFeatures, aggregatedResults, aggregatedBags, aggregatedUrlParts);
end

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

