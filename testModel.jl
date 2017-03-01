push!(LOAD_PATH, "EduNets/src");

using EduNets
using Plots

function testModelROC(model::AbstractModel, dataset::AbstractDataset)
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

function testModelPR(model::AbstractModel, dataset::AbstractDataset)
	out = forward!(model, dataset);
	pmask = dataset.y .== 2; # Bool array, true when dataset.y == 2 i. e. for real positives
	nmask = dataset.y .== 1;
	thresholds = sort(out[nmask], rev = true);
	precision = zeros(sum(nmask)); # Zero array with length equal to number of real negatives
	recall = zeros(sum(nmask));
	for (i, it) in enumerate(thresholds)
		precision[i] = mean(dataset.y[out .> it] .== 2)
		recall[i] = mean(out[pmask] .> it);
	end
	plot(recall, precision; xlabel = "Recall", ylabel = "Precision", xlims = (0, 1), ylims = (0, 1), label = "PR Curve");
end

