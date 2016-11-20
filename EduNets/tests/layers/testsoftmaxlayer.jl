using EduNets

function testsoftmaxlayer()
  nitems=100;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nitems);
  k=size(x,1)

  first=EduNets.SoftmaxLayer(k);
  gfirst=EduNets.SoftmaxLayer(k);
  second=EduNets.LinearLayer(k);
  gsecond=EduNets.LinearLayer(k);
  loss=EduNets.HingeLoss()

  function optFun(xx)
      O1=EduNets.forward!(first,xx);
      O2=EduNets.forward!(second,O1);
      (f,gO2)=EduNets.gradient!(loss,O2,y);
      gO1=EduNets.backprop!(second,O1,gO2,gsecond)
      gX=EduNets.backprop!(first,xx,gO1,gfirst)
      return(f,gX)
  end
  EduNets.testgradient(optFun,x;verbose=1);
end

testsoftmaxlayer()