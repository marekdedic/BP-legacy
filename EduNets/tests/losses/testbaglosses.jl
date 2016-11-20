using EduNets

function test(loss;T::DataType=Float64)
  x=randn(T,10,3);
  y=randn(T,10,3);
  bags=[1:3]

  f1=forward!(loss,x,bags,y,bags)
  (f2,gX)=gradient!(loss,x,bags,y,bags)
  println("equality of forward! and gradient! $(f1-f2)")
  f3=forward!(MeanL2BagLoss(T=T),x,bags,y,bags)
end

function test1(loss;T::DataType=Float64)
  x=randn(T,10,1);
  y=randn(T,10,1);
  bags=[1:1]

  function optFun(xx)
    (f,gX)=gradient!(loss,reshape(xx,size(y)),bags,y,bags)
    return(f,reshape(gX,length(gX)))
  end
  EduNets.testgradient(optFun,reshape(x,length(x));verbose=1,h=1e-6);
end

function test2(loss;T::DataType=Float64)
  x=randn(T,10,3);
  y=randn(T,10,3);
  bags=[1:3]

  function optFun(xx)
    (f,gX)=gradient!(loss,reshape(xx,size(y)),bags,y,bags)
    return(f,reshape(gX,length(gX)))
  end
  EduNets.testgradient(optFun,reshape(x,length(x));verbose=1,h=1e-6);
end

function test3(loss;T::DataType=Float64)
  x=randn(T,10,100);
  y=randn(T,10,100);
  ids=rand(1:10,100)
  xbags=map(i->findin(ids,i),1:10)
  ids=rand(1:10,100)
  ybags=map(i->findin(ids,i),1:10)

  function optFun(xx)
    (f,gX)=gradient!(loss,reshape(xx,size(y)),xbags,y,ybags)
    return(f,reshape(gX,length(gX)))
  end
  EduNets.testgradient(optFun,reshape(x,length(x));verbose=1,h=1e-6);
end

# T=Float64;
# for loss in [KLDivYXBagLoss(T=T),KLDivYXBagLoss(2,T=T),KLDivXYBagLoss(T=T),KLDivXYBagLoss(2,T=T),GaussMMDBagLoss(T=T),PairwiseL2BagLoss(T=T),HausdorffBagLoss(T=T),MeanL2BagLoss(T=T)]
#   println("testing ",typeof(loss))
#   test(loss)
#   test1(loss)
#   test2(loss)
#   test3(loss)
#   println()
# end

loss=KLDivYXBagLoss(T=T);
test3(loss);
Profile.clear()
@profile test3(loss);