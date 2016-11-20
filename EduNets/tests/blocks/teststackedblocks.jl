using EduNets

#tests the back_propagation
function test21(;T::DataType=Float64)
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,3),T=T);
  second=EduNets.ReluLayer((3,1),T=T);
  model=StackedBlocks(first,second,T=T);
  g=deepcopy(model)
  loss=EduNets.HingeLoss(T=T)
  init!(model,ds.x)

  function optFun(xx)
    O=forward!(model,xx)
    (f,gO)=EduNets.gradient(loss,O,ds.y);
    gX=EduNets.backprop!(model,xx,gO,g);
    return(f,gX)
  end
  EduNets.testgradient(optFun,ds.x;verbose=1,h=1e-3);
end

function test22(;T::DataType=Float64)
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,3),T=T);
  second=EduNets.ReluLayer((3,1),T=T);
  model=StackedBlocks(first,second,T=T);
  # init!(model,ds.x)
  g=deepcopy(model)
  loss=EduNets.HingeLoss(T=T)


  function optFun(x)
    update!(model,x)
    O=forward!(model,ds.x)
    (f,gO)=EduNets.gradient(loss,O,ds.y);
    EduNets.backprop!(model,ds.x,gO,g);
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,model2vector(model);verbose=1,h=1e-3);
end


function test23(;T::DataType=Float64)
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,3),T=T);
  second=EduNets.ReluLayer((3,1),T=T);
  model=StackedBlocks(first,second,T=T);
  g=deepcopy(model)
  loss=EduNets.HingeLoss(T=T)

  function optFun(x)
    update!(model,x)
    O=forward!(model,ds.x)
    (f,gO)=EduNets.gradient(loss,O,ds.y);
    EduNets.gradient!(model,ds.x,gO,g);
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,model2vector(model);verbose=1,h=1e-3);
end


#tests the back_propagation
function test31(;T::DataType=Float64)
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,3),T=T);
  second=EduNets.ReluLayer((3,4),T=T);
  third=EduNets.LinearLayer((4,1),T=T)
  model=StackedBlocks(first,second,third,T=T);
  g=deepcopy(model)
  loss=EduNets.HingeLoss(T=T)

  function optFun(xx)
    O=forward!(model,xx)
    (f,gO)=EduNets.gradient!(loss,O,ds.y);
    gX=EduNets.backprop!(model,xx,gO,g);
    return(f,gX)
  end
  EduNets.testgradient(optFun,ds.x;verbose=1,h=1e-3);
end

function test32(;T::DataType=Float64)
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,3),T=T);
  second=EduNets.ReluLayer((3,4),T=T);
  third=EduNets.LinearLayer((4,1),T=T)
  model=StackedBlocks(first,second,third,T=T);
  g=deepcopy(model)
  loss=EduNets.HingeLoss(T=T)

  function optFun(x)
    update!(model,x)
    O=forward!(model,ds.x)
    (f,gO)=EduNets.gradient!(loss,O,ds.y);
    EduNets.backprop!(model,ds.x,gO,g);
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,model2vector(model);verbose=1,h=1e-3);
end


function test33(;T::DataType=Float64)
  x=map(T,randn(5,30));
  y=sample([1,2],30);
  ds=EduNets.Dataset(x,y)

  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,3),T=T);
  second=EduNets.ReluLayer((3,4),T=T);
  third=EduNets.LinearLayer((4,1),T=T)
  model=StackedBlocks(first,second,third,T=T);
  g=deepcopy(model)
  loss=EduNets.HingeLoss(T=T)

  function optFun(x)
    update!(model,x)
    O=forward!(model,ds.x)
    (f,gO)=EduNets.gradient!(loss,O,ds.y);
    EduNets.gradient!(model,ds.x,gO,g);
    return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,model2vector(model);verbose=1,h=1e-3);
end


test21(;T=Float64)
test22(;T=Float64)
test23(;T=Float64)
test31(;T=Float64)
test32(;T=Float64)
test33(;T=Float64)
# testrelulayer2(;T=Float64)
# testrelulayer3(;T=Float64)
# testrelulayer4(;T=Float64)
# testrelulayer5(;T=Float64)

# testrelulayer1(;T=Float32)
# testrelulayer2(;T=Float32)
# testrelulayer3(;T=Float32)
# testrelulayer4(;T=Float32)
# testrelulayer5(;T=Float32)
