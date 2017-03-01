push!(LOAD_PATH, "EduNets/src");

using EduNets

function initializeModelUrl(;file::AbstractString="dataset.jld", percentage::Float32 = 0.9f0);
	insideLayers = 10;
	dataset = loadDataset(file);
	indices = randperm(length(dataset.labels));
	numTrain = Int(round(length(indices) * percentage));
	trainDataset = dataset[indices[1:numTrain]];
	testDataset = dataset[(numTrain + 1):length(indices)];
	startLayers= featureSize(trainDataset);
	model = UrlModel((ReluLayer((startLayers, insideLayers); T = Float32), MeanPoolingLayer(insideLayers; T = Float32)), (ReluLayer((startLayers, insideLayers); T = Float32), MeanPoolingLayer(insideLayers; T = Float32)), (ReluLayer((startLayers, insideLayers); T = Float32), MeanPoolingLayer(insideLayers; T = Float32)), (LinearLayer((3 * insideLayers, 1); T=Float32),), HingeLoss(;T = Float32));
	return (model, trainDataset, testDataset);
end

