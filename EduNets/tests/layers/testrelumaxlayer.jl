using EduNets

# function preparedataset(d,nbags,nitems;T::DataType=Float64)
#   x=map(T,randn(d,nitems));
#   ids=sample(1:nbags,nitems);
#   (bags,bagy)=EduNets.simplybag(ids,ones(nitems));
#   y=sample([1,2],length(bags));
#   return(x,y,bags)
# end

function findranges(ids)
  bags=fill(0:0,length(unique(ids)))
  idx=1
  bidx=1
  for i in 2:length(ids)
    if ids[i]!=ids[idx]
      bags[bidx]=idx:i-1
      idx=i;
      bidx+=1;
    end
  end
  if bidx<=length(bags)
    bags[bidx]=idx:length(ids)
  end
  return(bags)
end

function preparedataset(d,nbags,nitems;T::DataType=Float64)
  x=map(T,randn(d,nitems));
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  bags=findranges(sort(ids));
  return(x,y,bags)
end


function testrelumaxlayer1(;T::DataType=Float64)
  println("testing backpropagation")
  (x,y,bags)=preparedataset(100,1000,10000;T=T);
  k=size(x,1)

  first=ReluMaxLayer((k,10),T=T);
  g1=ReluMaxLayer((k,10),T=T);
  second=LinearLayer((10,1),T=T);
  g2=LinearLayer((10,1),T=T);
  loss=HingeLoss(T=T)

  function optFun(xx)
      O1=forward!(first,xx,bags);
      O2=forward!(second,O1);
      (f,gO2)=gradient!(loss,O2,y);
      gO1=backprop!(second,O1,gO2,g2)
      gX=backprop!(first,xx,gO1,g1)
      return(f,gX)
  end

  EduNets.testgradient(optFun,x;verbose=1,h=1e-3);
end

function testrelumaxlayer2(;T::DataType=Float64)
  # println("testing gradient in backpropagation")
  (fmat,y,bags)=preparedataset(10,10,100;T=T);
  k=size(fmat,1)

  first=ReluMaxLayer((k,10),T=T);
  g1=ReluMaxLayer((k,10),T=T);
  second=LinearLayer((10,1),T=T);
  g2=LinearLayer((10,1),T=T);
  loss=HingeLoss(T=T)

  function optFun(x)
    update!(first,x);
    O1=forward!(first,fmat,bags);
    O2=forward!(second,O1);
    (f,gO2)=gradient!(loss,O2,y);
    gO1=backprop!(second,O1,gO2,g2)
    gX=backprop!(first,fmat,gO1,g1)
    return(f,model2vector(g1))
  end

  EduNets.testgradient(optFun,model2vector(first);verbose=0,h=1e-3);
end



function testrelumaxlayer3(;T::DataType=Float64)
  println("testing gradient")
  (fmat,y,bags)=preparedataset(10,10,100;T=T);
  k=size(fmat,1)

  first=ReluMaxLayer((k,10),T=T);
  g1=ReluMaxLayer((k,10),T=T);
  second=LinearLayer((10,1),T=T);
  g2=LinearLayer((10,1),T=T);
  loss=HingeLoss(T=T)

  function optFun(x)
    update!(first,x);
    O1=forward!(first,fmat,bags);
    O2=forward!(second,O1);
    (f,gO2)=gradient!(loss,O2,y);
    gO1=backprop!(second,O1,gO2,g2)
    gradient!(first,fmat,gO1,g1)
    return(f,model2vector(g1))
  end

  EduNets.testgradient(optFun,model2vector(first);verbose=1,h=1e-3);
end


# testrelumaxlayer1(;T=Float64);
# testrelumaxlayer2(;T=Float64);
# testrelumaxlayer3(;T=Float64);

# testrelumaxlayer1(;T=Float32);
testrelumaxlayer2(;T=Float32);
@time for i in 1:1000 testrelumaxlayer2(;T=Float32); end;
# testrelumaxlayer3(;T=Float32);

