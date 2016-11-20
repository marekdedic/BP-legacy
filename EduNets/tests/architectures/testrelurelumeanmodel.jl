using EduNets
include("../../examples/relurelumean.jl")
function createdataset()
  d=10;
  ni=10000;
  nb=100;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  y=sample(1:2,nb);

  ds=SingleBagDataset(x,y,ids);
end

function testrelurelumeanmodel1()
  ds=createdataset();

  k=(size(ds.x,1),5,5)
  model=ReluReluMeanModel(k;T=Float64);
  g=ReluReluMeanModel(k;T=Float64);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

testrelurelumeanmodel1()
