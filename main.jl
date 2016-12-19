push!(LOAD_PATH, "EduNets/src");

include("include.jl");

(model, dataset) = initializeModel();
trainModel!(model, dataset);

testDataset = loadDataset("testDataset.jld");

#testModelROCCustom(model, dataset);
#testModelROC(model, dataset);

