push!(LOAD_PATH, "EduNets/src");

include("include.jl");

(model, trainDataset, testDataset) = initializeModel(file = "datasetFull.jld", percentage = 0.86f0);
trainModel!(model, trainDataset);

#testModelROC(model, testDataset);
#testModelPR(model, testDataset);

