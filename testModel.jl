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
	modC = deepcopy(model);
	datC = deepcopy(dataset);
	out = forward!(modC, datC);
	samples = 100;
	TPR = Array{Float32, 1}(samples);
	FPR = Array{Float32, 1}(samples);
	for i in 1:samples
		clamped = threshold(out, Float32(i / samples));
		(truePositiveRate, falsePositiveRate) = TPRFPR(clamped, datC.y);
		push!(TPR, truePositiveRate);
		push!(FPR, falsePositiveRate);
	end
	plot(TPR, FPR);
end

function testModelROC(model::SingleBagModel, dataset::SingleBagDataset)
	modC = deepcopy(model);
	datC = deepcopy(dataset);
	out = forward!(modC, datC);
	(target, nontarget) = separate(out, datC.y);
	rocPlot = roc(target, nontarget)
	Winston.plot(rocPlot);
end

function testModelPR(model::SingleBagModel, dataset::SingleBagDataset)
	modC = deepcopy(model);
	datC = deepcopy(dataset);
	out = forward!(modC, datC);
	samples = 100;
	P = Array{Float32, 1}(samples);
	R = Array{Float32, 1}(samples);
	for i in 1:samples
		clamped = threshold(out, Float32(i / samples));
		(precision, recall) = PR(clamped, datC.y);
		push!(P, precision);
		push!(R, recall);
	end
	plot(P, R);
end
