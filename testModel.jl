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
	plot(FPR, TPR; xlabel = "False-positive rate", ylabel = "True-positive rate", xlims = (0, 1), ylims = (0, 1), label = "ROC Curve");
	plot!(identity; linestyle = :dot, label="");
end

function testModelROC(model::SingleBagModel, dataset::SingleBagDataset)
	out = forward!(model, dataset);
	rocPlot = roc(out[dataset.y .== 2], out[dataset.y .!= 2])
	Winston.plot(rocPlot);
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
		Precision[i] = mean(dataset.y[out .> it] .== 2)
	end
	plot(Recall, Precision; xlabel = "Recall", ylabel = "Precision", xlims = (0, 1), ylims = (0, 1), label = "PR Curve");
end
