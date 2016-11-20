using EduNets
include("../../examples/doublebag.jl")
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

function test1()
  ds=createdataset();

  model=DoubleBagModel(ReluLayer((size(ds.x,1),7);T=Float64),
    MeanPoolingLayer(7;T=Float64),
    LinearLayer((7,4)),
    MaxPoolingLayer(4;T=Float64),
    LinearLayer((4,1);T=Float64),
    HingeLoss(;T=Float64));
  g=deepcopy(model);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

function test3()
  ds=createdataset();

  model=DoubleBagModel(ReluLayer((size(ds.x,1),7);T=Float64),
    # MeanPoolingLayer(7;T=Float64),
    ReluMaxLayer((7,7);T=Float64),
    LinearLayer((7,4)),
    MeanPoolingLayer(4;T=Float64),
    LinearLayer((4,1);T=Float64),
    HingeLoss(;T=Float64));
  g=deepcopy(model);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

function test4()
  ds=createdataset();

  model=DoubleBagModel(VoidLayer(size(ds.x,1);T=Float64),
    MeanPoolingLayer(10;T=Float64),
    VoidLayer(10;T=Float64),
    ReluMaxLayer((10,10);T=Float64),
    LinearLayer((10,1);T=Float64),
    HingeLoss(;T=Float64));
  g=deepcopy(model);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

function test2()
  ds=createdataset();

  model=DoubleBagModel(VoidLayer(size(ds.x,1),T=Float64),
    ReluMaxLayer((size(ds.x,1),7);T=Float64),
    VoidLayer(7,T=Float64),
    ReluMaxLayer((7,4);T=Float64),
    LinearLayer((4,1);T=Float64),
    HingeLoss(;T=Float64));
  g=deepcopy(model);
  theta=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=gradient!(model,ds,g)
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

test1()
test2()
test3()
test4()
