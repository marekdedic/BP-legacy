using EduNets
function testsoftmaxloss()
  x=randn(5,10);
  y=sample(1:5,10);

  loss=SoftmaxLoss(abs(randn(5)));

  function optFun(xx::Matrix)
    (f,g)=gradient!(loss,xx,y)
    return(f,g)
  end
  EduNets.testgradient(optFun,x;verbose=1)
end

testsoftmaxloss()