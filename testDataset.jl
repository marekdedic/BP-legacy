using EduNets;

function testDataset(features::Matrix, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; info::Vector{AbstractString} = Vector{AbstractString}(0), T::DataType = Float32)
	if(!issorted(urlIDs))
		permutation = sortperm(urlIDs);
		features = features[:, permutation];
		labels = labels[permutation];
		urlIDs = urlIDs[permutation];
		urlParts = urlParts[permutation];
		if size(info, 1) != 0;
			info = info[permutation];
		end
	end
	subbags = findranges(urlIDs);

	domainFeatures = Vector{Vector{T}}(0);
	pathFeatures = Vector{Vector{T}}(0);
	queryFeatures = Vector{Vector{T}}(0);
	bagLabels = Vector{Int}(length(subbags));
	if size(info, 1) != 0;
		bagInfo = Vector{AbstractString}(length(subbags));
	else
		bagInfo = Vector{AbstractString}(0);
	end
	# TODO: Implement bags
	bags = Vector{UnitRange{Int}}(length(subbags));

	for (i, r) in enumerate(subbags)
		for (j, part) in enumerate(urlParts[r])
			if part == 1
				push!(domainFeatures, features[:, first(r) + j - 1]);
			elseif part == 2
				push!(pathFeatures, features[:, first(r) + j - 1]);
			elseif part == 3
				push!(queryFeatures, features[:, first(r) + j - 1]);
			end
		end
		bagLabels[i] = maximum(labels[r]);
		if size(info, 1) != 0;
			bagInfo[i] = info[r][1];
		end
		bags[i] = i:i;
	end

	return (domainFeatures, pathFeatures, queryFeatures, bagLabels, convert(DataFrames.DataFrame, reshape(bagInfo, length(bagInfo), 1)))
end


function testOne()
	fVec = rand(0:1, 256);
	f = hcat([fVec for i in 1:1000]...) 
	l = rand(1:2, 1000);
	uId = rand(1:200, 1000);
	sort!(uId);
	uPar = rand(1:3, 1000);
	for i in 1:100
		dRes = maximum(mapslices(std, hcat(testDataset(f, l, uId, uPar)[1]...), 2)) == 0.0;
		pRes = maximum(mapslices(std, hcat(testDataset(f, l, uId, uPar)[2]...), 2)) == 0.0;
		qRes = maximum(mapslices(std, hcat(testDataset(f, l, uId, uPar)[3]...), 2)) == 0.0;
		if !dRes || !pRes || !qRes;
			return false;
		end
	end
	return true;
end

function test()
	for i in 1:100
		if(!testOne())
			return false;
		end
	end
	return true;
end
