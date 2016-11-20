using EduNets
include("../../examples/relumaxrelumax.jl")
function createdataset()
  d=10;
  ni=10000;
  nb=100;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  subbags=map(i->find(ids.==i),unique(ids));
  ids=sample(1:nb,length(subbags));
  bags=map(i->find(ids.==i),unique(ids));
  y=sample(1:4,length(bags));

  ds=DoubleBagDataset(x,y,bags,subbags,DataFrames.DataFrame([]));
end

function testrelumaxrelumaxmodel1()
  ds=createdataset();

  k=(size(ds.x,1),5,5)
  model=ReluMaxReluMaxModel(k;T=Float64);
  g=ReluMaxReluMaxModel(k;T=Float64);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

function testrelumaxrelumaxmodel3()
  ds=createdataset()

  k=(size(ds.x,1),5,5,maximum(ds.y))
  model=ReluMaxReluMaxModel(k;T=Float64);
  g=ReluMaxReluMaxModel(k;T=Float64);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=EduNets.gradient!(model,ds,g)
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

testrelumaxrelumaxmodel1()
testrelumaxrelumaxmodel3()
