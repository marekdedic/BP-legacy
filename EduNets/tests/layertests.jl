
function testrelumaxlayer()
  x=randn(10,1000);
  y=sample([1,2],100);
  ids=sample(1:100,1000);
  ds=EduNets.SingleBagDataset(x,y,ids);

  k=(size(ds.x,1),1)
  model=EduNets.SingleReluMaxModel(k;T=Float64);
  g=EduNets.SingleReluMaxModel(k;T=Float64);
  theta=EduNets.model2vector(model)
  function optFun(x::Vector)
    EduNets.update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    # (f,g)=EduNets.gradient(ds,model)
    return(f,EduNets.model2vector(g))
  end
  @time EduNets.testgradient(optFun,theta)
end

function testlqpoolinglayerBP()
  x=randn(10,1000);
  y=sample([1,2],100);
  ids=sample(1:100,1000);
  ds=EduNets.SingleBagDataset(x,y,ids);

  k=(size(ds.x,1),1)
  model=EduNets.SingleReluMaxModel(k;T=Float64);
  g=EduNets.SingleReluMaxModel(k;T=Float64);
  theta=EduNets.model2vector(model)
  function optFun(x::Vector)
    EduNets.update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    # (f,g)=EduNets.gradient(ds,model)
    return(f,EduNets.model2vector(g))
  end
  @time EduNets.testgradient(optFun,theta)
end


function testurlrelulayer()
  d=20;
  indexes=rand(1:d,100); #features are in the range of 1 to 20
  sampleindexes=rand(1:10,100); #we will have 10 samples (maximum, each having approximately 100 samples)
  y=sample([1,2],10); #randomly generate the labels

  k=(d,1)
  model=EduNets.SingleURLReluModel(k;T=Float64);
  g=EduNets.SingleURLReluModel(k;T=Float64);
  theta=EduNets.model2vector(model)
  function optFun(x::Vector)
    EduNets.update!(model,x)
    # f=EduNets.gradient!(ds,model,g)
    (f,g)=EduNets.gradient(indexes,sampleindexes,y,model)
    return(f,EduNets.model2vector(g))
  end
  @time EduNets.testgradient(optFun,theta)
end