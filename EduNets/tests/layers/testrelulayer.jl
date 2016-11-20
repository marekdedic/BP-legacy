using EduNets

function preparedataset(T)
  l=100
  d=10
  k=10;
  x=map(T,randn(d,l));
  y=sample([1,2],l);

  return(EduNets.Dataset(x,y),k)
end

function testrelulayer3(;T::DataType=Float64)
  (ds,k)=preparedataset(T)
  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,k),T=T);
  second=EduNets.ReluLayer((k,1),T=T);
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
  EduNets.testgradient(optFun,ds.x;verbose=1,h=1e-6);
end

function testrelulayer4(;T::DataType=Float64)
  (ds,k)=preparedataset(T)
  d=size(ds.x,1)
  first=EduNets.ReluLayer((d,k),T=T);
  second=EduNets.ReluLayer((k,1),T=T);
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
    EduNets.gradient!(first,ds.x,gO1,gfirst)

    offset=EduNets.model2vector!(gfirst,g);
    EduNets.model2vector!(gsecond,g;offset=offset);
    return(f,g)
  end
  EduNets.testgradient(optFun,theta;verbose=1,h=1e-6);
end

# @time testrelulayer3(;T::DataType=Float64)
# Profile.clear_malloc_data();
testrelulayer3(;T=Float64)
testrelulayer4(;T=Float64)
# @time testrelulayer3(;T=Float64)
# @time begin
  # for i in 1:100
    # testrelulayer3(;T=Float64)
  # end
# end
# testrelulayer4(;T=Float64)
# testrelulayer5(;T=Float64)

# testrelulayer3(;T=Float32)
# testrelulayer4(;T=Float32)
# testrelulayer5(;T=Float32)
