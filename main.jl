push!(LOAD_PATH, "EduNets/src");

include("include.jl");

(model, trainDataset, testDataset) = initializeModel(file = "dataset.jld", percentage = 0.86f0);
trainModel!(model, trainDataset);

#testModelROCCustom(model, testDataset);
#testModelPR(model, testDataset);

