using EduNets
include("/Users/tpevny/Work/julia/EduNets/lib/examples/concoder.jl")


function test1()
  nitems=100;
  d=10;
  xx=randn(d,nitems);
  model=ConCoder(ReluLayer((d,5)),ReluLayer((5,d)));
  g=deepcopy(model)
  function optFun(x::Vector)
      update!(model,x)
      f=gradient!(model,xx,g)
      return(f,model2vector(g))
  end
  # optFun(theta)
  EduNets.testgradient(optFun,model2vector(model);verbose=1)
end

function test2()
  nitems=100;
  d=10;
  xx=randn(d,nitems);
  model=ConCoder(StackedBlocks(ReluLayer((d,7)),ReluLayer((7,5))),StackedBlocks(ReluLayer((5,7)),ReluLayer((7,d))));
  g=deepcopy(model)
  function optFun(x::Vector)
      update!(model,x)
      f=gradient!(model,xx,g)
      return(f,model2vector(g))
  end
  EduNets.testgradient(optFun,model2vector(model);verbose=1)
end
test1()
test2()
