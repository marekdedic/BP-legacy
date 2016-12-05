push!(LOAD_PATH, "EduNets/src");

include("parseDataset.jl");
include("initializeModel.jl");
include("trainModel.jl");

parseTGSubsample() = parseDataset("../threatGridSamples2/0");

