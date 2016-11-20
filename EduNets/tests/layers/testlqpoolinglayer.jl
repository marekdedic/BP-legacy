using EduNets

function testlqpoolinglayer1()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=LqPoolingLayer(k);
  gfirst=LqPoolingLayer(k);
  second=LinearLayer(k);
  gsecond=LinearLayer(k);
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

function testlqpoolinglayer2()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=LqPoolingLayer(k);
  gfirst=LqPoolingLayer(k);
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


testlqpoolinglayer1()
testlqpoolinglayer2()