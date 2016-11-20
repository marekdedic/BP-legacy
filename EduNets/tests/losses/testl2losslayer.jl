#tests the back_propagation
using EduNets

function testl2loss1()
  println("testing the backpropagation of linear layer")
  x=randn(5,30);
  y=randn(5,30);
  loss=L2Loss()

  function optFun(xx)
    (f,g)=EduNets.gradient!(loss,xx,y);
    return(f,g)
  end
  EduNets.testgradient(optFun,x;verbose=1);
end

testl2loss1()