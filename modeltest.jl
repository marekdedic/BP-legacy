include("UrlDataset.jl")
include("UrlModel.jl")
include("processDataset.jl")
include("featureGenerators.jl")

function randurl()
	builder = Vector{String}(0);
	push!(builder, "http");
	if rand(Bool)
		push!(builder, "s");
	end
	push!(builder, "://");
	if rand(Bool)
		push!(builder, "www.");
	end
	push!(builder, randstring(rand(1:100)));
	for i in 1:rand(0:10)
		push!(builder, ".");
		push!(builder, randstring(rand(1:100)));
	end
	push!(builder, "/");
	if rand(Bool)
		push!(builder, randstring(rand(1:100)));
		for i in 1:rand(0:10)
			push!(builder, "/");
			push!(builder, randstring(rand(1:100)));
		end
		if rand(Bool)
			push!(builder, ".");
			push!(builder, randstring(rand(1:10)));
		end
	end
	if rand(Bool)
		push!(builder, "?");
		push!(builder, randstring(rand(1:30)));
		push!(builder, "=");
		push!(builder, randstring(rand(1:30)));
		for i in 1:rand(0:10)
			push!(builder, "&");
			push!(builder, randstring(rand(1:30)));
			push!(builder, "=");
			push!(builder, randstring(rand(1:30)));
		end
	end
	return join(builder)
end

function testModel(model::UrlModel)
	dataset = [randurl() for i in 1:rand(1:10)];
	input = processDataset(dataset, rand(1:2, size(dataset)));
	reference = forward!(model, input)[end][end, :];
	for i in 1:100
		if forward!(model, input)[end][end, :] != reference
			return false;
		end
	end
	return true;
end

function test()
	k = 20;
	d = 2053;
	o = 2;
	T = Float32;
	model = UrlModel((ReluLayer((d, k); T = T), MeanPoolingLayer(k; T = T)),
					 (ReluLayer((d, k); T = T), MeanPoolingLayer(k; T = T)),
					 (ReluLayer((d, k); T = T), MeanPoolingLayer(k; T = T)),
					 (LinearLayer((3 * k,o); T=T),));
	for i in 1:1000
		if !testModel(model)
			return false;
		end
	end
	return true;
end
