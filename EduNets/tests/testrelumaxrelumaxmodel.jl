using EduNets
import Base.size,Base.length;
include("../../architectures/doublebag.jl")
function testrelumaxrelumaxmodel()
  testrelumaxrelumaxmodel1()
  testrelumaxrelumaxmodel2()
  testrelumaxrelumaxmodel3()
  nothing
end


function testrelumaxrelumaxmodel1()
  d=10;
  ni=10000;
  nb=100;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  subbags=map(i->find(ids.==i),unique(ids));
  ids=sample(1:nb,length(subbags));
  bags=map(i->find(ids.==i),unique(ids));
  y=sample([1,2],length(bags));

  ds=EduNets.DoubleBagDataset(x,y,bags,subbags,DataFrames.DataFrame([]));

  k=(size(ds.x,1),5,5,1)
  model=EduNets.ReluMaxReluMaxModel(k,HingeLoss([0.5,0.5]);T=Float64);
  g=EduNets.ReluMaxReluMaxModel(k,HingeLoss([0.5,0.5]);T=Float64);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

function testrelumaxrelumaxmodel2()
  d=10;
  ni=10000;
  nb=100;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  subbags=map(i->find(ids.==i),unique(ids));
  ids=sample(1:nb,length(subbags));
  bags=map(i->find(ids.==i),unique(ids));
  y=sample([1,2],length(bags));

  ds=EduNets.DoubleBagDataset(x,y,bags,subbags,DataFrames.DataFrame([]));

  k=(size(ds.x,1),5,5,1)
  model=EduNets.ReluMaxReluMaxModel(k,HingeLoss([0.5,0.5]);T=Float64);
  function optFun(x::Vector)
    update!(model,x)
    (f,g)=EduNets.gradient(ds,model)
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,model2vector(model);verbose=1)
end


function testrelumaxrelumaxmodel3()
  d=10;
  ni=10000;
  nb=100;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  subbags=map(i->find(ids.==i),unique(ids));
  ids=sample(1:nb,length(subbags));
  bags=map(i->find(ids.==i),unique(ids));
  y=sample(1:4,length(bags));

  ds=EduNets.DoubleBagDataset(x,y,bags,subbags,DataFrames.DataFrame([]));

  k=(size(ds.x,1),5,5,maximum(y))
  model=EduNets.ReluMaxReluMaxModel(k,HingeLoss([0.5,0.5]);T=Float64);
  g=EduNets.ReluMaxReluMaxModel(k,HingeLoss([0.5,0.5]);T=Float64);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end