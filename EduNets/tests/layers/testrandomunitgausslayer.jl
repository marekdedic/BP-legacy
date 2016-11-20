using EduNets
#tests the back_propagation
function testrandomgauss1(;T::DataType=Float64)
  println("testing the backpropagation of RandomUnitGaussLayer layer")
  x=map(T,randn(1,1));
  first=EduNets.RandomUnitGaussLayer(T=T);
  gfirst=EduNets.RandomUnitGaussLayer(T=T);

  function optFun(x)
    srand(1)
    O=forward!(first,x);
    f=O[1]
    gX=backprop!(first,x,ones(T,1,1),gfirst)
    return(f,gX)
  end
  EduNets.testgradient(optFun,x;verbose=1,h=1e-3);
end

function testrandomgauss2(;T::DataType=Float64)
  println("testing the backpropagation of RandomUnitGaussLayer layer")
  x=map(T,randn(5,100));
  y=rand(1:2,100)
  first=EduNets.RandomUnitGaussLayer(T=T);
  second=EduNets.LinearLayer((5,1),T=T);
  gfirst=EduNets.RandomUnitGaussLayer(T=T);
  gsecond=EduNets.LinearLayer((5,1),T=T);
  loss=HingeLoss(T=T);

  function optFun(x)
    srand(1)
    O1=forward!(first,x);
    O2=forward!(second,O1);
    (f,gO2)=gradient!(loss,O2,y)
    gO1=backprop!(second,O1,gO2,gsecond)
    gX=backprop!(first,x,gO1,gfirst)
    return(f,gX)
  end
  EduNets.testgradient(optFun,x;verbose=1,h=1e-3);
end


function testrandomgauss3(;T::DataType=Float64)
  println("testing the backpropagation of RandomUnitGaussLayer layer")
  x=map(T,randn(5,10));
  y=rand(1:2,100)
  first=EduNets.RandomUnitGaussLayer(T=T);
  second=EduNets.LinearLayer((5,1),T=T);
  gfirst=EduNets.RandomUnitGaussLayer(T=T);
  gsecond=EduNets.LinearLayer((5,1),T=T);
  loss=HingeLoss(T=T);

  function optFun(x)
    srand(1)
    (O1,bags)=forward!(first,x,fill(10,10));
    O2=forward!(second,O1);
    (f,gO2)=gradient!(loss,O2,y)
    gO1=backprop!(second,O1,gO2,gsecond)
    gX=backprop!(first,x,gO1,bags,gfirst)
    return(f,gX)
  end
  EduNets.testgradient(optFun,x;verbose=1,h=1e-3);
end


testrandomgauss1()
testrandomgauss2()
testrandomgauss3()