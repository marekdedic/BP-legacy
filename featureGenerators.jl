"Generates an array of all the n-grams (substrings of length n) from a given string."
function ngrams(input::AbstractString, n::Int)::Vector{AbstractString}
	output = fill("", max(length(input) - n + 1, 0));
	for (i, c) in enumerate(input)
		for j in (i - n + 1):i
			if j > 0 && j <= length(output)
				output[j] = string(output[j], c);
			end
		end
	end
	return output;
end

unigrams(input::AbstractString)::Vector{AbstractString} = ngrams(input, 1)
bigrams(input::AbstractString)::Vector{AbstractString} = ngrams(input, 2)
trigrams(input::AbstractString)::Vector{AbstractString} = ngrams(input, 3);

function trigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32)::Vector{Float32}
	output = zeros(T, modulo);
	for i in trigrams(input)
		index = mod(hash(i), modulo);
		output[index + 1] += 1;
	end
	return output;
end
