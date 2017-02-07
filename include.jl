push!(LOAD_PATH, "EduNets/src");

include("UrlDatasetCompound.jl");

include("parseDataset.jl");
include("initializeModel.jl");
include("initializeModelUrl.jl");
include("trainModel.jl");
include("testModel.jl");

