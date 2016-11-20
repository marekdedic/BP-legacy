
function testwithadam()
  x=map(Float64,randn(10,10000));
  ids=sample(1:100,10000);
  subbags=map(i->find(ids.==i),unique(ids));
  ids=sample(1:100,length(subbags));
  bags=map(i->find(ids.==i),unique(ids));
  y=sample([1,2],length(bags));
  ds=EduNets.DoubleBagDataset(x,y,bags,subbags,DataFrames.DataFrame([]));

  k=(size(ds.x,1),5,5)
  model=EduNets.ReluMaxReluMaxModel(k;T=Float64);
  g=EduNets.ReluMaxReluMaxModel(k;T=Float64);
  function optFun(x::Vector)
    EduNets.update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    # (f,g)=EduNets.gradient(ds,model)
    println()
    println(EduNets.model2vector(g))
    return(f,EduNets.model2vector(g))
  end
  EduNets.testgradient(optFun,EduNets.model2vector(model);verbose=1)
  EduNets.adam(optFun,EduNets.model2vector(model),EduNets.AdamOptions(;maxIter=1000))
end
