using EduNets
include("../../examples/singlebag.jl")
function createdataset()
  d=10;
  ni=10000;
  nb=100;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  y=sample(1:2,nb);

  ds=SingleBagDataset(x,y,ids);
end

function test()
  ds=createdataset();

  model=SingleBagModel(ReluLayer((size(ds.x,1),5);T=Float64),MeanPoolingLayer(5;T=Float64),LinearLayer((5,1);T=Float64),HingeLoss(;T=Float64));
  g=deepcopy(model);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

test()
