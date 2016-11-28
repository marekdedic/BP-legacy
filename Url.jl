push!(LOAD_PATH, "EduNets/src");

import GZip
import JSON
import EduNets

type Url
	protocol::AbstractString
	domain::AbstractArray{AbstractString, 1};
	port::Int64;
	path::AbstractArray{AbstractString, 1};
	query::AbstractArray{Tuple{AbstractString, AbstractString}, 1};
end;
Url() = Url("", [], 80, [], []);

function readURL(input::AbstractString)::Url
  output::Url = Url();
  splitted = split(input, "://");
  output.protocol = splitted[1];
  if in(':', splitted[2])
	  splitted = split(splitted[2], ":");
	  domain = splitted[1];
	  splitted = split(splitted[2], "/");
	  output.port = parse(Int64, splitted[1]);
	  splice!(splitted, 1);
  end
  query = split(splitted[end], "?");
  splitted[end] = query[1];
  query = query[2];
  for s in split(domain, ".")
	  push!(output.domain, s);
  end
  for s in splitted
	  push!(output.path, s);
  end
  for s in split(query, "&")
	  keyvalue = split(s, "=");
	  push!(output.query, (keyvalue[1], keyvalue[2]));
  end
  println(output);
  return output;
end

