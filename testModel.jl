push!(LOAD_PATH, "EduNets/src");

using EduNets
using ROCAnalysis, Winston, MLPlots

function separate(predicted::StridedArray{Float32}, real::Array{Int64})::Tuple{Array{Float32}, Array{Float32}}
	target = Array{Float32, 1}(0);
	nontarget = Array{Float32, 1}(0);
	for i in 1:length(real)
		if(real[i] == 2)
			push!(target, predicted[i]);
		else
			push!(nontarget, predicted[i]);
		end
	end
	return (target, nontarget);
end

function testModelROC(model::SingleBagModel, dataset::SingleBagDataset)
	out = forward!(model, dataset);
	(target, nontarget) = separate(out, dataset.y);
	rocPlot = roc(target, nontarget);
	Winston.plot(rocPlot);
end

