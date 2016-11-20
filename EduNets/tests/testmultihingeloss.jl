function testmultihingeloss()
  x=randn(5,100);
  y=sample(1:5,100);
  W=abs(randn(5))

  loss=MultiHingeLoss(W);

  function optFun(xx::Matrix)
    (f,g)=gradient!(loss,xx,y)
    return(f,g)
  end
  EduNets.testgradient(optFun,x;verbose=1)

  function optFun(xx::Matrix)
    (f,g)=gradient!(loss,xx,y)
    return(f,g)
  end
  EduNets.testgradient(optFun,x;verbose=1)
  nothing
end