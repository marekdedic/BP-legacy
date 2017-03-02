push!(LOAD_PATH, "EduNets/src");

include("UrlDataset.jl");
include("UrlModel.jl");

include("loadDataset.jl");
include("initializeModel.jl");
include("initializeModelUrl.jl");
include("trainModel.jl");
include("trainModelUrl.jl");

