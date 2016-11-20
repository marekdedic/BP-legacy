using StatsBase
function testrelumean1()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);

  k=(size(ds.x,1),5)
  model=EduNets.ReluMeanModel(k;T=Float64);
  g=EduNets.ReluMeanModel(k;T=Float64);
  theta=EduNets.model2vector(model)
  function optFun(x::Vector)
    EduNets.update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

testrelumean1()
