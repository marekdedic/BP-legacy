using EduNets

function test1()
  theta=randn(100);
  y=sample([1,2],100);

  loss=PrecisionAtLoss(0.5);
  function optFun(x::Vector)
    (f,g)=gradient!(loss,x',y)
    return(f,g')
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

test1();