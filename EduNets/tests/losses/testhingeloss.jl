using EduNets

function testhingeloss1()
  theta=randn(100);
  y=sample([1,2],100);

  loss=HingeLoss([0.1,0.9]);
  function optFun(x::Vector)
    (f,g)=gradient!(loss,x',y)
    return(f,g')
  end
  EduNets.testgradient(optFun,theta;verbose=1)

  function optFun(x::Vector)
    (f,g)=gradient(loss,x',y)
    return(f,g')
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

function testhingeloss2()
  theta=map(Float32,randn(100));
  # y=map(Int32,sample([1,2],100));
  y=sample([1,2],100);

  loss=HingeLoss([0.1f0,0.9f0]);
  function optFun(x)
    (f,g)=gradient!(loss,x',y)
    return(f,g')
  end
  EduNets.testgradient(optFun,theta;verbose=1,h=1e-3)

  function optFun(x::Vector{Float32})
    (f,g)=gradient(loss,x',y)
    return(f,g')
  end
  EduNets.testgradient(optFun,theta;verbose=1,h=1e-3)
end

testhingeloss1();
testhingeloss2();