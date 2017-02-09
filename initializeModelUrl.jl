push!(LOAD_PATH, "EduNets/src");

import JLD
using EduNets

function loadDataset(file::AbstractString)
	return JLD.load(file, "dataset");
end

function initializeModelUrl(;file::AbstractString="dataset.jld", percentage::Float32 = 0.9f0);
	insideLayers = 100;
	dataset = loadDataset(file);
	indices = randperm(length(dataset.labels));
	numTrain = Int(round(length(indices) * percentage));
	trainDataset = dataset[indices[1:numTrain]];
	testDataset = dataset[(numTrain + 1):length(indices)];
	startLayers= featureSize(trainDataset);
	model = UrlModelCompound((ReluLayer((startLayers, insideLayers); T = Float32), MeanPoolingLayer(insideLayers; T = Float32)), (ReluLayer((startLayers, insideLayers); T = Float32), MeanPoolingLayer(insideLayers; T = Float32)), (ReluLayer((startLayers, insideLayers); T = Float32), MeanPoolingLayer(insideLayers; T = Float32)), (LinearLayer((insideLayers, 1); T=Float32),));
	loss = HingeLoss(; T = Float32);
	return (model, loss, trainDataset, testDataset);
end

