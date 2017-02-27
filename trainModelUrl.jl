push!(LOAD_PATH, "EduNets/src");

using EduNets

function trainModelUrl!(model::UrlModelCompound, dataset::UrlDatasetCompound; T::DataType=Float32, lambda::Float32=1f-6, iter::Int=1000)::Void
  gg=model2vector(model);
  g=deepcopy(model)

  function optFun(x::Vector)
    update!(model,x);
	dss = dataset;
    #dss = sample(dataset,[1000,1000]);

	f = fgradient!(model, dss, g);

	model2vector!(g, gg);

	(f,gg) #return function value and gradient of the error
  end

  theta=model2vector(model);
  adam(optFun, theta, AdamOptions(;maxIter = iter));
  return nothing;
end
