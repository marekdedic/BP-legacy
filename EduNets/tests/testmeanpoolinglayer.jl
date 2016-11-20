using StatsBase;

function testmeanpoolinglayer1()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=EduNets.MeanPoolingLayer(k);
  second=EduNets.LinearLayer(k);
  loss=EduNets.HingeLoss()

  function optFun(xx)
      O1=EduNets.forward(xx,first,ds.bags);
      O2=EduNets.forward(O1,second);
      (f,gO2)=EduNets.gradient(loss,O2,ds.y);
      (g2,gO1)=EduNets.backprop(O1,second,gO2)
      gX=EduNets.backprop(xx,first,ds.bags,gO1)
      return(f,gX)
  end
  EduNets.testgradient(optFun,ds.x;verbose=1);
end

function testmeanpoolinglayer2()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=EduNets.MeanPoolingLayer(k);
  second=EduNets.LinearLayer(k);
  gsecond=EduNets.LinearLayer(k);
  loss=EduNets.HingeLoss()

  function optFun(xx)
      O1=EduNets.forward!(xx,first,ds.bags);
      O2=EduNets.forward!(O1,second);
      (f,gO2)=EduNets.gradient(loss,O2,ds.y);
      gO1=EduNets.backprop!(O1,second,gO2,gsecond)
      gX=EduNets.backprop!(xx,first,ds.bags,gO1)
      return(f,gX)
  end
  EduNets.testgradient(optFun,ds.x;verbose=1);
end
testmeanpoolinglayer1()
testmeanpoolinglayer2()