import GZip;
import JSON;
import JLD;
import DataFrames;
using EduNets;

include("UrlDataset.jl");
include("featureGenerators.jl");
include("labellers.jl");
include("processDataset.jl");
include("IterableParser.jl");

# Helper functions

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

# ThreatgridSample loading functions

function loadThreatGrid(dir::AbstractString; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, labeller::Function = countingLabeller, T::DataType = Float32)::SingleBagDataset
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
				result = labeller(filename * ".vt.json");
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

function loadThreatGridUrl(dir::AbstractString; batchSize::Int = 6000, featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, labeller::Function = countingLabeller, T::DataType = Float32)::IterableParser
	urls = Vector{Vector{AbstractString}};
	labels = Vector{Int};
	for (root, dirs, files) in walkdir(dir)
		for file in filter(x-> ismatch(r"\.joy\.json\.gz$", x), files)
			path = joinpath(root, file);
			filename = replace(path, r"^(.*)\.joy\.json\.gz$", s"\1");
			if isfile(filename * ".vt.json")
				push!(urls, loadUrlFromJSON(filename * ".joy.json.gz"));
				push!(labels, labeller(filename * ".vt.json"));
			end
		end
	end
	return IterableParser(vcat(urls...), labels, batchSize; featureCount = featureCount, featureGenerator = featureGenerator, T = T)
end

# Sample loading function
function loadSampleUrl(file::AbstractString; batchSize::Int = 6000, featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)::IterableParser
	table = GZip.open(file,"r") do fid
		readcsv(fid)
	end
	if any(table[:, 3].!="legit")
		table=table[table[:, 3].!="legit",:]
	end

	urls = table[:, 1];
	labels = (table[:, 3].!="legit")+1;
	return IterableParser(convert(Vector{AbstractString}, urls), convert(Vector{Int}, labels), batchSize; featureCount = featureCount, featureGenerator = featureGenerator, T = T)
end

# Actual realisations of a complete dataset parser.

function parseThreatGrid(dir::AbstractString, file::AbstractString = "dataset.jld")::Void
	JLD.save(file, "dataset", loadThreatGrid(dir));
	return nothing;
end

function parseThreatGridAVClass(dir::AbstractString, file::AbstractString = "dataset.jld")::Void
	setupAVClass("avclass_results.txt");
	dataset = loadThreatGrid(dir, labeller = AVClassLabeller);
	JLD.save(file, "dataset", dataset);
	return nothing;
end

function parseThreatGridAVClassUrl(dir::AbstractString, file::AbstractString = "dataset.jld")::Void
	setupAVClass("avclass_results.txt");
	for i in loadThreatGridUrl(dir, labeller = AVClassLabeller; batchSize = 0);
		JLD.save(file, "dataset", i);
	end
	return nothing;
end

function parseSampleUrl(source::AbstractString, file::AbstractString = "dataset.jld")::Void
	for i in loadSampleUrl(source; batchSize = 0)
		JLD.save(file, "dataset", i);
	end
	return nothing;
end

