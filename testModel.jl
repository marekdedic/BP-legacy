push!(LOAD_PATH, "EduNets/src");

using EduNets

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

function testModel(model::SingleBagModel, dataset::SingleBagDataset)
	out = forward!(model, dataset);
	clamped = threshold(out, 0.5f0);
	return clamped;
end
