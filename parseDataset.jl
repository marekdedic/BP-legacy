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

function features(input::AbstractString, modulo::Int64=2053)::SparseVector{Float32}
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

function loadResultFromJSON(file::AbstractString)::Int64
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

function loadThreatGrid(dir::AbstractString)::SingleBagDataset
	featureMatrix = [Array{Float32, 2}(2053, 0) for i in 1:Threads.nthreads()];
	results = [Array{Int64}(0) for i in 1:Threads.nthreads()];
	bags = [Array{Int}(0) for i in 1:Threads.nthreads()];
	maxBag = [1 for i in 1:Threads.nthreads()];
	aggregatedFeatures = Array{Float32, 2}(2053, 0);
	aggregatedResults = Array{Int64}(0);
	aggregatedBags = Array{Int}(0);
	for (root, dirs, files) in walkdir(dir)
		Threads.@threads for file in filter(x-> ismatch(r"\.joy\.json\.gz$", x), files)
			path = joinpath(root, file);
			filename = replace(path, r"^(.*)\.joy\.json\.gz$", s"\1");
			if isfile(filename * ".vt.json")
				urls = loadUrlFromJSON(filename * ".joy.json.gz");
				result = loadResultFromJSON(filename * ".vt.json");
				for url in urls
					featureMatrix[Threads.threadid()] = hcat(featureMatrix[Threads.threadid()], features(url));
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

function parseDataset(dir::AbstractString, file::AbstractString="dataset.jld")::Void
	JLD.save(file, "dataset", loadThreatGrid(dir));
	return nothing;
end
