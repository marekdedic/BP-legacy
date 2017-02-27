push!(LOAD_PATH, "EduNets/src");

include("include.jl");

(model, trainDataset, testDataset) = initializeModelUrl(file = "datasetCompound.jld", percentage = 0.86f0);
trainModelUrl!(model, trainDataset);

#testModelROC(model, testDataset);
#testModelPR(model, testDataset);

