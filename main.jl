push!(LOAD_PATH, "EduNets/src");

include("include.jl");

(model, trainDataset, testDataset) = initializeModel();
trainModel!(model, trainDataset);

#testModelROCCustom(model, testDataset);
#testModelPR(model, testDataset);

