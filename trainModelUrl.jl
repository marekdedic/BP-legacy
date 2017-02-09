push!(LOAD_PATH, "EduNets/src");

using EduNets

function trainModelUrl!(model::UrlModelCompound, loss::AbstractLoss, dataset::UrlDatasetCompound; T::DataType=Float32, lambda::Float32=1f-6, iter::Int=1000)::Void
  #sc=ScalingLayer(dataset.x);
  g=deepcopy(model);
  gg=model2vector(model);

  function optFun(x::Vector)
    update!(model,x);
    #dss = sample(dataset,[1000,1000]);
    f = gradient!(model,dss,g);
    #f += l1regularize!(model,g, T(lambda));
    model2vector!(g,gg);
    return(f,gg);
  end

  theta=model2vector(model);
  adam(optFun, theta, AdamOptions(;maxIter=iter));
  return nothing;
end