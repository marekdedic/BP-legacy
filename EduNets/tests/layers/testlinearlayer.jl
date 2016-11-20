using EduNets
#tests the back_propagation
function testlinearlayer1(;T::DataType=Float64)
  println("testing the backpropagation of linear layer")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.LinearLayer(d,3,T=T);
  second=EduNets.LinearLayer(3,1,T=T);
  loss=EduNets.HingeLoss(T=T)

  function optFun(xx)
    O1=EduNets.forward(first,xx);
    O2=EduNets.forward(second,O1);
    (f,gO2)=EduNets.gradient(loss,O2,ds.y);
    (g2,gO1)=EduNets.backprop(second,O1,gO2)
    (g1,gX)=EduNets.backprop(first,xx,gO1)
    return(f,gX)
  end
  EduNets.testgradient(optFun,ds.x;verbose=1,h=1e-3);
end


#tests the gradient
function testlinearlayer2(;T::DataType=Float64)
  println("testing the gradient in the backpropagation")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.LinearLayer(d,3,T=T);
  second=EduNets.LinearLayer(3,1,T=T);
  loss=EduNets.HingeLoss(T=T)

  theta=vcat(EduNets.model2vector(first),EduNets.model2vector(second));
  function optFun(x)
    offset=EduNets.update!(first,x);
    EduNets.update!(second,x;offset=offset);
    O1=EduNets.forward(first,ds.x);
    O2=EduNets.forward(second,O1);
    (f,gO2)=EduNets.gradient(loss,O2,ds.y);
    (g2,gO1)=EduNets.backprop(second,O1,gO2)
    (g1,gX)=EduNets.backprop(first,ds.x,gO1)
    g=vcat(EduNets.model2vector(g1),EduNets.model2vector(g2));
    return(f,g)
  end
  EduNets.testgradient(optFun,theta;verbose=1,h=1e-3);
end


function testlinearlayer3(;T::DataType=Float64)
  println("testing the overwriting backpropagation")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.LinearLayer(d,3,T=T);
  second=EduNets.LinearLayer(3,1,T=T);
  loss=EduNets.HingeLoss(T=T)

  gfirst=deepcopy(first);
  gsecond=deepcopy(second);
  
  function optFun(xx)
    O1=EduNets.forward!(first,xx);
    O2=EduNets.forward!(second,O1);
    (f,gO2)=EduNets.gradient!(loss,O2,ds.y);
    gO1=EduNets.backprop!(second,O1,gO2,gsecond)
    gX=EduNets.backprop!(first,xx,gO1,gfirst)
    return(f,gX)
  end
  EduNets.testgradient(optFun,ds.x;verbose=1,h=1e-3);
end

function testlinearlayer4(;T::DataType=Float64)
  println("testing the gradient in the  backpropagation")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.LinearLayer(d,3,T=T);
  second=EduNets.LinearLayer(3,1,T=T);
  loss=EduNets.HingeLoss(T=T)
  theta=vcat(EduNets.model2vector(first),EduNets.model2vector(second))

  gfirst=deepcopy(first);
  gsecond=deepcopy(second);
  g=deepcopy(theta)
  
  function optFun(x)
    offset=EduNets.update!(first,x);
    EduNets.update!(second,x;offset=offset);
    O1=EduNets.forward!(first,ds.x);
    O2=EduNets.forward!(second,O1);
    (f,gO2)=EduNets.gradient!(loss,O2,ds.y);
    gO1=EduNets.backprop!(second,O1,gO2,gsecond)
    gX=EduNets.backprop!(first,ds.x,gO1,gfirst)
    offset=EduNets.model2vector!(gfirst,g);
    EduNets.model2vector!(gsecond,g;offset=offset);
    return(f,g)
  end
  EduNets.testgradient(optFun,theta;verbose=1,h=1e-3);
end


testlinearlayer1()
testlinearlayer2()
testlinearlayer3()
testlinearlayer4()

testlinearlayer1(T=Float32)
testlinearlayer2(T=Float32)
testlinearlayer3(T=Float32)
testlinearlayer4(T=Float32)