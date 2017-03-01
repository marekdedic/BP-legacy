push!(LOAD_PATH, "EduNets/src");

import JLD
using EduNets

function loadDataset(file::AbstractString)
	return JLD.load(file, "dataset");
end

