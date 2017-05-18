"Sets up a global dictionary for AVClassLabeller, so that it doesn't have to be loaded every time."
function setupAVClass()
	global avclass_dict = Dict{String, String}();
	run(`./avclass_proccess.sh`);
	open("avclass_results.txt") do file
		for line in eachline(file)
			values = split(chomp(line), '\t');
			if size(values)[1] > 2
				avclass_dict[values[1]] = values[3];
			end
		end
	end
	run(`./avclass_results.txt`);
end

function AVClassLabeller(file::AbstractString)::Int
	result = get(avclass_dict, file, "");
	return (result == "CLEAN" || result == "") ? 1 : 2;
end
