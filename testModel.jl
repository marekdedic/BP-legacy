push!(LOAD_PATH, "EduNets/src");

using EduNets
using ROCAnalysis
using MLPlots
import Winston

function testModelROCCustom(model::SingleBagModel, dataset::SingleBagDataset)
	out = forward!(model, dataset);
	pmask = dataset.y .== 2; # Bool array, true when dataset.y == 2
	nmask = dataset.y .== 1;
	thresholds = sort(out[nmask], rev = true);
	TPR = zeros(sum(nmask)); # Zero array with length equal to number of real negatives
	FPR = zeros(sum(nmask));
	for (i,it) in enumerate(thresholds)
		FPR[i] = mean(out[nmask] .> it); # Mean over Bool array (percentage of true values), true when truly negative but prediction higher then threshold
		TPR[i] = mean(out[pmask] .> it);
	end
	plot(FPR, TPR; xlabel = "False-positive rate", ylabel = "True-positive rate");
end

function testModelROC(model::SingleBagModel, dataset::SingleBagDataset)
	out = forward!(model, dataset);
	rocPlot = roc(predicted[real .== 2], predicted[real .!= 2])
	plot(rocPlot);
end

function testModelPR(model::SingleBagModel, dataset::SingleBagDataset)
	out = forward!(model, dataset);
	pmask = dataset.y .== 2; # Bool array, true when dataset.y == 2
	nmask = dataset.y .== 1;
	thresholds = sort(out[nmask], rev = true);
	Precision = zeros(sum(nmask)); # Zero array with length equal to number of real negatives
	Recall = zeros(sum(nmask));
	for (i, it) in enumerate(thresholds)
		Recall[i] = mean(out[pmask] .> it);
		ppmask = out .> it;
		Precision[i] = mean(dataset.y[ppmask] .== 2)
	end
	plot(Recall, Precision; xlabel = "Recall", ylabel = "Precision");
end
