using EduNets
include("../../examples/relurelumax.jl")
function createdataset()
  d=10;
  ni=10000;
  nb=100;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  y=sample(1:2,nb);

  ds=SingleBagDataset(x,y,ids);
end

function testrelurelumaxmodel1()
  ds=createdataset();

  k=(size(ds.x,1),5,5)
  model=ReluReluMaxModel(k;T=Float64);
  g=ReluReluMaxModel(k;T=Float64);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

testrelurelumaxmodel1()
