using EduNets

function testgausspoolinglayer1()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=GaussPoolingLayer(k;n=10);
  gfirst=deepcopy(first);
  second=LinearLayer(k);
  gsecond=deepcopy(second);
  loss=HingeLoss()

  function optFun(xx)
    xx=reshape(xx,size(ds.x))

    O1=forward!(first,xx,ds.bags);
    O2=forward!(second,O1);
    (f,gO2)=gradient!(loss,O2,ds.y);
    gO1=backprop!(second,O1,gO2,gsecond)
    gX=backprop!(first,xx,ds.bags,gO1,gfirst)
    return(f,reshape(gX,length(ds.x)))
  end
  testgradient(optFun,reshape(ds.x,length(x));verbose=1);
end

function testgausspoolinglayer11()
  nitems=1;
  x=randn(1,nitems);
  first=GaussPoolingLayer(1;n=10);
  gfirst=deepcopy(first);

  function optFun(xx)
    xx=reshape(xx,size(x))

    bags=[1:size(xx,2)]
    O1=forward!(first,xx,bags);
    f=O1[1]
    gO1=ones(1,1)
    gX=backprop!(first,xx,bags,gO1,gfirst)
    return(f,reshape(gX,length(x)))
  end
  testgradient(optFun,reshape(x,length(x));verbose=1);
end

function testgausspoolinglayer21()
  nitems=1;
  x=randn(1,nitems);
  first=GaussPoolingLayer(1;n=2);
  gfirst=deepcopy(first);

  function optFun(xx)
    update!(first,xx)

    bags=[1:size(xx,2)]
    O1=forward!(first,x,bags);
    f=O1[1]
    gO1=ones(1,1)
    gX=backprop!(first,x,bags,gO1,gfirst)
    return(f,model2vector(gfirst))
  end
  testgradient(optFun,model2vector(first);verbose=1);
end

function testgausspoolinglayer2()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=GaussPoolingLayer(k,n=10);
  init!(first,x)
  gfirst=GaussPoolingLayer(k,n=10);
  second=LinearLayer(k);
  gsecond=LinearLayer(k);
  loss=HingeLoss()

  gq=model2vector(gfirst)

  function optFun(q)
      # update!(first,1+log(1+exp(q)))
      update!(first,q)
      O1=forward!(first,ds.x,ds.bags);
      O2=forward!(second,O1);
      (f,gO2)=gradient!(loss,O2,ds.y);
      gO1=backprop!(second,O1,gO2,gsecond)
      backprop!(first,ds.x,ds.bags,gO1,gfirst)
      model2vector!(gfirst,gq);
      return(f,gq)
  end

  testgradient(optFun,model2vector(first);verbose=1);
end


testgausspoolinglayer11()
testgausspoolinglayer21()
testgausspoolinglayer1()
testgausspoolinglayer2()