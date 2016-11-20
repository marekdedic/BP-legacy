function testhingeloss()
  theta=randn(100);
  y=sample([1,2],100);

  loss=HingeLoss();
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

function testequalhingeloss()
  theta=randn(100);
  y=sample([1,2],100);

  loss=EqualHingeLoss();
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

function testlogisticloss()
  theta=randn(100);
  y=sample([1,2],100);

  loss=LogisticLoss();
  function optFun(x::Vector)
    (f,g)=backprop!(loss,x',y)
    return(f,g')
  end
  @time EduNets.testgradient(optFun,theta)
end
