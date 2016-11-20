using EduNets
include("../../examples/colinda.jl")

function testcolinda1()
  xx=randn(10,100)

  k=(10,5)
  model=CoLinDa(k;T=Float64);
  g=deepcopy(model)
  theta=model2vector(model)
  theta=EduNets.model2vector(model)
  function optFun(x::Vector)
    EduNets.update!(model,x)
    f=gradient!(model,xx,g)
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end

testcolinda1()
