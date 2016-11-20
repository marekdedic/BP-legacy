using EduNets
#tests the back_propagation
function test1(;T::DataType=Float64)
  println("testing the backpropagation of linear layer")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=Dataset(x,y)

  d=size(ds.x,1)
  first=Dropout(LinearLayer((d,3),T=T),0.8);
  second=Dropout(LinearLayer((3,1),T=T),0.5);
  loss=HingeLoss(T=T)

  function optFun(xx)
    O1=forward(first,xx);
    O2=forward(second,O1);
    (f,gO2)=gradient(loss,O2,ds.y);
    (g2,gO1)=backprop(second,O1,gO2)
    (g1,gX)=backprop(first,xx,gO1)
    return(f,gX)
  end
  testgradient(optFun,ds.x;verbose=1,h=1e-3);
end


#tests the gradient
function test2(;T::DataType=Float64)
  println("testing the gradient in the backpropagation")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=Dataset(x,y)

  d=size(ds.x,1)
  first=Dropout(LinearLayer((d,3),T=T),0.8);
  second=Dropout(LinearLayer((3,1),T=T),0.5);
  loss=HingeLoss(T=T)

  theta=vcat(model2vector(first),model2vector(second));
  function optFun(x)
    offset=update!(first,x);
    update!(second,x;offset=offset);
    O1=forward(first,ds.x);
    O2=forward(second,O1);
    (f,gO2)=gradient(loss,O2,ds.y);
    (g2,gO1)=backprop(second,O1,gO2)
    (g1,gX)=backprop(first,ds.x,gO1)
    g=vcat(model2vector(g1),model2vector(g2));
    return(f,g)
  end
  testgradient(optFun,theta;verbose=1,h=1e-3);
end


function test3(;T::DataType=Float64)
  println("testing the overwriting backpropagation")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=Dataset(x,y)

  d=size(ds.x,1)
  first=Dropout(LinearLayer((d,3),T=T),0.8);
  second=Dropout(LinearLayer((3,1),T=T),0.5);
  loss=HingeLoss(T=T)

  gfirst=deepcopy(first);
  gsecond=deepcopy(second);
  
  function optFun(xx)
    O1=forward!(first,xx);
    O2=forward!(second,O1);
    (f,gO2)=gradient!(loss,O2,ds.y);
    gO1=backprop!(second,O1,gO2,gsecond)
    gX=backprop!(first,xx,gO1,gfirst)
    return(f,gX)
  end
  testgradient(optFun,ds.x;verbose=1,h=1e-3);
end

function test4(;T::DataType=Float64)
  println("testing the gradient in the  backpropagation")
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=Dataset(x,y)

  d=size(ds.x,1)
  first=Dropout(LinearLayer((d,3),T=T),0.8);
  second=Dropout(LinearLayer((3,1),T=T),0.5);
  loss=HingeLoss(T=T)
  theta=vcat(model2vector(first),model2vector(second))

  gfirst=deepcopy(first);
  gsecond=deepcopy(second);
  g=deepcopy(theta)
  
  function optFun(x)
    offset=update!(first,x);
    update!(second,x;offset=offset);
    O1=forward!(first,ds.x);
    O2=forward!(second,O1);
    (f,gO2)=gradient!(loss,O2,ds.y);
    gO1=backprop!(second,O1,gO2,gsecond)
    gX=backprop!(first,ds.x,gO1,gfirst)
    offset=model2vector!(gfirst,g);
    model2vector!(gsecond,g;offset=offset);
    return(f,g)
  end
  testgradient(optFun,theta;verbose=1,h=1e-3);
end


test1()
test2()
test3()
test4()

test1(T=Float32)
test2(T=Float32)
test3(T=Float32)
test4(T=Float32)