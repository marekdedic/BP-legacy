function testrelulqrelulqmodel()
  x=map(Float64,randn(10,10000));

  ids=sample(1:100,10000);
  subbags=map(i->find(ids.==i),unique(ids));
  ids=sample(1:100,length(subbags));
  bags=map(i->find(ids.==i),unique(ids));
  y=sample([1,2],length(bags));

  ds=EduNets.DoubleBagDataset(x,y,bags,subbags,DataFrames.DataFrame([]));

  k=(size(ds.x,1),5,5)
  model=EduNets.ReluLqReluLqModel(k;T=Float64);

  g=EduNets.ReluLqReluLqModel(k;T=Float64);
  bds=cumsum([0,length(model.first),length(model.firstpool),length(model.second),length(model.secondpool)])
  idxs=collect(bds[2]+1:bds[3])
  idxs=append!(idxs,bds[4]+1:bds[5])

  theta=EduNets.model2vector(model)
  gg=EduNets.model2vector(model)

  function optFun(x::Vector)
    for i in idxs
      x[i]=1+log(1+exp(x[i]));
    end
    EduNets.update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    EduNets.model2vector!(g,gg)
    for i in idxs
      x[i]=log(exp(x[i]-1)-1)
      gg[i]*=exp(x[i])/(1+exp(x[i]))
    end
    return(f,gg)
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end


function testrelumaxmodel()
  x=randn(10,1000);
  y=sample([1,2],100);
  ids=sample(1:100,1000);
  ds=SingleBagDataset(x,y,ids);

  k=(size(ds.x,1),5)
  model=ReluMaxModel(k;T=Float64);
  g=ReluMaxModel(k;T=Float64);
  theta=model2vector(model)
  gg=model2vector(model)
  function optFun(x::Vector)
    update!(model,x)
    f=EduNets.gradient!(ds,model,g)
    EduNets.model2vector!(g,gg);
    return(f,gg)
  end
  EduNets.testgradient(optFun,theta;verbose=1)
end