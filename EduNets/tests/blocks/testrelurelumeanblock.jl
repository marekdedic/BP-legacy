using StatsBase
using EduNets

""" test of the gradient with non-cached forward"""
function testrelurelumax1()
  nitems=100;
  nbags=10;
  d=10;
  x=sparse(randn(d,nitems));
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);

  k=(size(ds.x,1),5,1)
  model=EduNets.ReluReluMeanBlock(k;T=Float64);
  g=EduNets.ReluReluMeanBlock(k;T=Float64);
  theta=EduNets.model2vector(model)
  loss=EduNets.HingeLoss()
  function optFun(x::Vector)
    EduNets.update!(model,x)
    o=EduNets.forward!(model,ds.x,ds.bags)
    (f,gO)=EduNets.gradient!(loss,o,ds.y)
    EduNets.gradient!(model,ds.x,gO,g);
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

""" test of the gradient with non-cached forward"""
function testrelurelumax2()
  nitems=100;
  nbags=10;
  d=10;
  x=sparse(randn(d,nitems));
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  o=zeros(1,10)

  k=(size(ds.x,1),5,1)
  model=EduNets.ReluReluMeanBlock(k;T=Float64);
  g=EduNets.ReluReluMeanBlock(k;T=Float64);
  theta=EduNets.model2vector(model)
  loss=EduNets.HingeLoss()
  function optFun(x::Vector)
    EduNets.update!(model,x)
    EduNets.forward!(model,ds.x,ds.bags,o)
    (f,gO)=EduNets.gradient!(loss,o,ds.y)
    EduNets.gradient!(model,ds.x,gO,g);
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

""" test of the gradient with non-cached forward"""
function testrelurelumax3()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  o=zeros(1,10)

  k=(size(ds.x,1),5,1)
  model=EduNets.ReluReluMeanBlock(k;T=Float64);
  g=EduNets.ReluReluMeanBlock(k;T=Float64);
  theta=EduNets.model2vector(model)
  loss=EduNets.HingeLoss()
  function optFun(x::Matrix)
    EduNets.forward!(model,x,ds.bags,o)
    (f,gO)=EduNets.gradient!(loss,o,ds.y)
    gX=EduNets.backprop!(model,x,gO,g);
    return(f,gX)
  end
  EduNets.testgradient(optFun,ds.x;verbose=1)
end

""" test of the gradient with non-cached forward"""
function testrelurelumax4()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  o=zeros(1,10)

  k=(size(ds.x,1),5,1)
  model=EduNets.ReluReluMeanBlock(k;T=Float64);
  g=EduNets.ReluReluMeanBlock(k;T=Float64);
  theta=EduNets.model2vector(model)
  loss=EduNets.HingeLoss()
  function optFun(x)
    EduNets.update!(model,x)
    EduNets.forward!(model,ds.x,ds.bags,o)
    (f,gO)=EduNets.gradient!(loss,o,ds.y)
    gX=EduNets.backprop!(model,ds.x,gO,g);
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

println("testing calculation of gradient of ReluReluMeanBlock without pre-allocation of outputs")
testrelurelumax1();

println("testing calculation of gradient of ReluReluMeanBlock with pre-allocation of outputs");
testrelurelumax2();

println("testing the back-propagation of ReluReluMeanBlock");
testrelurelumax3();

println("testing the gradient in back-propagation of ReluReluMeanBlock");
testrelurelumax4();
