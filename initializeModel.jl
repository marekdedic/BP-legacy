push!(LOAD_PATH, "EduNets/src");

import JLD
using EduNets

function initializeModel(file::AbstractString="dataset.jld")
	insideLayers = 100;
	dataset = JLD.load("dataset.jld", "dataset");
	model = SingleBagModel(StackedBlocks(ReluLayer((size(dataset.x, 1), insideLayers); T=Float32), ReluLayer((insideLayers, insideLayers); T=Float32);T=Float32), MeanPoolingLayer(insideLayers; T=Float32), LinearLayer((insideLayers, 1); T=Float32), HingeLoss(; T=Float32); T=Float32);
	return (model, dataset);
end

