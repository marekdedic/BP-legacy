using EduNets
include("/Users/tpevny/Work/julia/EduNets/lib/examples/milautoencoder.jl")


function test1()
 nitems=100;
  nbags=10;
  d=10;
  xx=randn(d,nitems);
  ids=sample(1:nbags,nitems);
  ds=SingleBagDataset(xx,zeros(Int,nbags),ids)
  y=randn(d,nbags);
  model=MilCoder(ReluLayer((d,5)),MaxPoolingLayer(5),ReluLayer((5,d)),L2Loss());
  g=deepcopy(model)
  function optFun(x::Vector)
      EduNets.update!(model,x)
      f=gradient!(model,ds.x,ds.bags,y,g)
      return(f,EduNets.model2vector(g))
  end
  # optFun(theta)
  EduNets.testgradient(optFun,model2vector(model);verbose=1)
end

function test2()
 nitems=100;
  nbags=10;
  d=10;
  xx=randn(d,nitems);
  ids=sample(1:nbags,nitems);
  ds=SingleBagDataset(xx,zeros(Int,nbags),ids)
  y=randn(d,nbags);
  model=MilCoder(VoidLayer(d),ReluMaxLayer((d,5)),ReluLayer((5,d)),L2Loss());
  g=deepcopy(model)
  function optFun(x::Vector)
      EduNets.update!(model,x)
      f=gradient!(model,ds.x,ds.bags,y,g)
      return(f,EduNets.model2vector(g))
  end
  # optFun(theta)
  EduNets.testgradient(optFun,model2vector(model);verbose=1)
end

test1()
test2()
