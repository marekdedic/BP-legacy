import JSON

type Url
	protocol::AbstractString
	domain::AbstractArray{AbstractString, 1};
	port::Int64;
	path::AbstractArray{AbstractString, 1};
	query::AbstractArray{Tuple{AbstractString, AbstractString}, 1};
end;
Url() = Url("", [], 80, [], []);

function readURL(input::AbstractString)
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

function trigrams(input::AbstractString)
	output::Array{AbstractString} = [];
	i::Int64 = 1;
	for c in input
		push!(output, AbstractString(""));
		output[i] = string(output[i], c);
		if i > 1
			output[i - 1] = string(output[i - 1], c);
		end
		if i > 2
			output[i - 2] = string(output[i - 2], c);
		end
		i += 1;
	end
	return output;
end

readURL("https://mojeweby.cz:8080/directory/index.php?user=guest&topic=main");

# function singletrain(filenames,model::EduNets.AbstractModel,scalingfile,oprefix;preprocess::Array{EduNets.AbstractModel,1}=Array{EduNets.AbstractModel,1}(0),lambda::Float32=1e-6,T::DataType=Float32)
#=function singletrain(filenames,model::EduNets.AbstractModel,coder::EduNets.AbstractModel,scalingfile,oprefix;preprocess::Array{EduNets.AbstractModel,1}=Array{EduNets.AbstractModel,1}(0),lambda::Float32=1e-6,T::DataType=Float32)
  negfiles=filter(x->contains(x,"neg.jld"),filenames)
  posfiles=filter(x->contains(x,"pos.jld"),filenames)
  sc=EduNets.ScalingLayer(scalingfile,T=T);
  g=deepcopy(model)
  gg=ReadData.model2vector(model);

  function loaddata(model,errfile)
    ds=ReadData.loadscattered(ReadData.samplefiles((negfiles,posfiles),[3,3]);T=T);
    EduNets.scale!(ds.x,sc);
    ds.x=EduNets.forward!(coder,ds.x);
    ds.y[ds.y.>1]=2;

    err=forward(model.loss,forward!(model,ds),ds.y);
    println("error on the fresh dataset");
    open(errfile,"a") do fid
      write(fid,@sprintf("%g\n",err))
    end
    return(ds)
  end

  function optFun(ds::EduNets.DoubleBagDataset,x::Vector)
    ReadData.update!(model,x)
    dss=EduNets.sample(ds,[1000,1000];subbagsize=100);
    f=ReadData.gradient!(model,dss,g)
    f+=ReadData.l1regularize!(model,g;lambda=T(lambda/length(x)));
    ReadData.model2vector!(g,gg)
    return(f,gg)
  end

  theta=ReadData.model2vector(model);
  EduNets.adam(()->loaddata(model,oprefix*".err"),optFun,theta;options=EduNets.AdamOptions(;maxIter=1000,progressFile=oprefix),numberofdataloads=300)
end=#
