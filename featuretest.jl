include("featureGenerators.jl")

function subtest(rounds::Int = 10)
	size = rand(1:10000);
	len = rand(1:10000);
	str = randstring(len);
	hash = trigramFeatureGenerator(str, size);
	for i in 1:rounds
		if trigramFeatureGenerator(str, size) != hash
			return false;
		end
	end
	return true;
end

function test()
	for i in 1:10000
		if(!subtest())
			return false;
		end
	end
	return true;
end
