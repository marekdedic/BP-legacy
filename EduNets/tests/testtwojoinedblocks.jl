using StatsBase

function createSimpleDataset()
 nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  return(EduNets.SingleBagDataset(x,y,ids));
end

""" test of the gradient with non-cached forward"""
function testtwojoinedblocks1()
  println("testing calculation of gradient of TwoJoinedBlocks")
  ds1=createSimpleDataset();
  ds2=createSimpleDataset();

  first=EduNets.ReluReluMaxBlock((size(ds1.x,1),5,1));
  second=EduNets.ReluReluMaxBlock((size(ds2.x,1),5,1));
  third=EduNets.ReluLayer(2,1);
  model=EduNets.TwoJoinedBlocks(first,second,);

  gmodel=deepcopy(model);
  gthird=deepcopy(third);

  loss=EduNets.HingeLoss()

  function optFun(x::Vector)
    EduNets.update!(model,x)
    o1=EduNets.forward!(model,ds1.x,ds1.bags,ds2.x,ds2.bags)
    o2=EduNets.forward!(third,o1)
    (f,go2)=EduNets.gradient!(loss,o2,ds1.y)
    go1=EduNets.backprop!(third,o1,go2,gthird)
    EduNets.gradient!(model,ds1.x,ds2.x,go1,gmodel)
    return(f,EduNets.model2vector(gmodel))
  end

  EduNets.testgradient(optFun,EduNets.model2vector(model);verbose=1)
end

""" test of the gradient with non-cached forward"""
function testtwojoinedblocks2()
  println("testing calculation of backpropagation of TwoJoinedBlocks")
  ds1=createSimpleDataset();
  ds2=createSimpleDataset();

  first=EduNets.ReluReluMaxBlock((size(ds1.x,1),5,1));
  second=EduNets.ReluReluMaxBlock((size(ds2.x,1),5,1));
  third=EduNets.ReluLayer(2,1);
  model=EduNets.TwoJoinedBlocks(first,second,);

  gmodel=deepcopy(model);
  gthird=deepcopy(third);

  loss=EduNets.HingeLoss()

  function optFun(x)
    o1=EduNets.forward!(model,x,ds1.bags,ds2.x,ds2.bags)
    o2=EduNets.forward!(third,o1)
    (f,go2)=EduNets.gradient!(loss,o2,ds1.y)
    go1=EduNets.backprop!(third,o1,go2,gthird)
    (gx1,gx2)=EduNets.backprop!(model,x,ds2.x,go1,gmodel)
     # backprop!(model::TwoJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,go,g::TwoJoinedBlocks)
    return(f,gx1)
  end

  EduNets.testgradient(optFun,ds1.x;verbose=1)

  function optFun(x)
    o1=EduNets.forward!(model,ds1.x,ds1.bags,x,ds2.bags)
    o2=EduNets.forward!(third,o1)
    (f,go2)=EduNets.gradient!(loss,o2,ds1.y)
    go1=EduNets.backprop!(third,o1,go2,gthird)
    (gx1,gx2)=EduNets.backprop!(model,ds1.x,x,go1,gmodel)
     # backprop!(model::TwoJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,go,g::TwoJoinedBlocks)
    return(f,gx2)
  end

  EduNets.testgradient(optFun,ds2.x;verbose=1)
end

testtwojoinedblocks1();
testtwojoinedblocks2();

# println("testing calculation of gradient of ReluReluMaxBlock with pre-allocation of outputs");
# testtwojoinedblocks2();

# println("testing the back-propagation of ReluReluMaxBlock");
# testtwojoinedblocks3();

# println("testing the gradient in back-propagation of ReluReluMaxBlock");
# testtwojoinedblocks4();
