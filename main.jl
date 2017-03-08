push!(LOAD_PATH, "EduNets/src");

include("include.jl");

(model, loss, trainDataset, testDataset) = initializeModelUrl(file = "dataset.jld", percentage = 0.86f0);
trainModelUrl!(model, loss, trainDataset);

#testModelROC(model, testDataset);
#testModelPR(model, testDataset);

