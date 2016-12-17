push!(LOAD_PATH, "EduNets/src");

using EduNets
using ROCAnalysis
using MLPlots
import Winston

function threshold(input::StridedArray{Float32}, threshold::Float32)::Array{Int64}
	output = Array{Int64, 1}(length(input));
	for i in 1:length(input)
		if(input[i] < threshold)
			output[i] = 1;
		else
			output[i] = 2;
		end
	end
	return output;
end

function separate(predicted::StridedArray{Float32}, real::Array{Int64})::Tuple{Array{Float32}, Array{Float32}}
	target = Array{Float64, 1}(0);
	nontarget = Array{Float64, 1}(0);
	for i in 1:length(real)
		if(real[i] == 2)
			push!(target, predicted[i]);
		else
			push!(nontarget, predicted[i]);
		end
	end
	return (target, nontarget);
end

function PR(predicted::Array{Int64}, real::Array{Int64})::Tuple{Float32, Float32}
	tp = 0;
	fp = 0;
	fn = 0;
	for i in 1:length(real)
		if(predicted[i] == 2 && real[i] == 2)
			tp += 1;
		end
		if(predicted[i] == 2 && real[i] == 1)
			fp += 1;
		end
		if(predicted[i] == 1 && real[i] == 2)
			fn += 1;
		end
	end
	return (tp / (tp + fp), tp / (tp + fn));
end

function TPRFPR(predicted::Array{Int64}, real::Array{Int64})::Tuple{Float32, Float32}
	tp = 0;
	fp = 0;
	tn = 0;
	fn = 0;
	for i in 1:length(real)
		if(predicted[i] == 2 && real[i] == 2)
			tp += 1;
		end
		if(predicted[i] == 2 && real[i] == 1)
			fp += 1;
		end
		if(predicted[i] == 1 && real[i] == 2)
			fn += 1;
		end
		if(predicted[i] == 1 && real[i] == 1)
			tn += 1;
		end
	end
	return (fp / (fp + tn), tp / (tp + fn));
end

function ROCValue(T::Float32, predicted::StridedArray{Float32}, real::Array{Int64})::Tuple{Float32, Float32}
	clamped = threshold(predicted, T);
	(truePositiveRate, falsePositiveRate) = TPRFPR(clamped, real);
	return (truePositiveRate, falsePositiveRate);
end

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
	(target, nontarget) = separate(out, dataset.y);
	rocPlot = roc(target, nontarget)
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
		ppmask = out .> it;
		Precision[i] = mean(dataset.y[ppmask] .== 2)
		Recall[i] = mean(out[pmask] .> it);
	end
	plot(Recall, Precision; xlabel = "Recall", ylabel = "Precision");
end
