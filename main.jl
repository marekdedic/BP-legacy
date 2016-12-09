push!(LOAD_PATH, "EduNets/src");

import ROCAnalysis
import Winston

include("include.jl");

(model, dataset) = initializeModel();
trainModel!(model, dataset);

testModel(model, dataset);

