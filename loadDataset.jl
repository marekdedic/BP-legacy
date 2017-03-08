import JLD
using EduNets

function loadDataset(file::AbstractString)
	return JLD.load(file, "dataset");
end

