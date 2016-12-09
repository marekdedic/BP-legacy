push!(LOAD_PATH, "EduNets/src");

using EduNets

function testModel(model::SingleBagModel, dataset::SingleBagDataset; T::DataType=Float32, lambda::Float32=1f-6, iter::Int=1000)
  sc=ScalingLayer(dataset.x);
  g=deepcopy(model);
  gg=model2vector(model);

  function optFun(x::Vector)
    update!(model,x);
    dss=sample(dataset,[10,10]);
    f=gradient!(model,dss,g);
    f+=l1regularize!(model,g, T(lambda));
    model2vector!(g,gg);
    return(f,gg);
  end

  theta=model2vector(model);
	return testgradient(optFun, theta; verbose=1);
end
