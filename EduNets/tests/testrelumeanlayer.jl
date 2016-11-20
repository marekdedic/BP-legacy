
function testrelumeanlayer1()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=EduNets.ReluMeanLayer(k,10);
  g1=EduNets.ReluMeanLayer(k,10);
  second=EduNets.LinearLayer(10,1);
  g2=EduNets.LinearLayer(10,1);
  loss=EduNets.HingeLoss()

  function optFun(xx)
      O1=EduNets.forward!(first,xx,ds.bags);
      O2=EduNets.forward!(second,O1);
      (f,gO2)=EduNets.gradient(loss,O2,ds.y);
      gO1=EduNets.backprop!(second,O1,gO2,g2)
      gX=EduNets.backprop!(first,xx,ds.bags,gO1,g1)
      return(f,gX)
  end

  EduNets.testgradient(optFun,ds.x;verbose=1);
end

function testrelumeanlayer2()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=EduNets.ReluMeanLayer(k,10);
  g1=EduNets.ReluMeanLayer(k,10);
  second=EduNets.LinearLayer(10,1);
  g2=EduNets.LinearLayer(10,1);
  loss=EduNets.HingeLoss()

  function optFun(x)
    EduNets.update!(first,x);
    O1=EduNets.forward!(first,ds.x,ds.bags);
    O2=EduNets.forward!(second,O1);
    (f,gO2)=EduNets.gradient(loss,O2,ds.y);
    gO1=EduNets.backprop!(second,O1,gO2,g2)
    gX=EduNets.backprop!(first,ds.x,ds.bags,gO1,g1)
    return(f,EduNets.model2vector(g1))
  end

  EduNets.testgradient(optFun,EduNets.model2vector(first);verbose=1);
end



function testrelumeanlayer3()
  nitems=100;
  nbags=10;
  d=10;
  x=randn(d,nitems);
  y=sample([1,2],nbags);
  ids=sample(1:nbags,nitems);
  ds=EduNets.SingleBagDataset(x,y,ids);
  k=size(ds.x,1)

  first=EduNets.ReluMeanLayer(k,10);
  g1=EduNets.ReluMeanLayer(k,10);
  second=EduNets.LinearLayer(10,1);
  g2=EduNets.LinearLayer(10,1);
  loss=EduNets.HingeLoss()

  function optFun(x)
    EduNets.update!(first,x);
    O1=EduNets.forward!(first,ds.x,ds.bags);
    O2=EduNets.forward!(second,O1);
    (f,gO2)=EduNets.gradient(loss,O2,ds.y);
    gO1=EduNets.backprop!(second,O1,gO2,g2)
    EduNets.gradient!(first,ds.x,ds.bags,gO1,g1)
    return(f,EduNets.model2vector(g1))
  end

  EduNets.testgradient(optFun,EduNets.model2vector(first);verbose=1);
end


testrelumeanlayer1()
testrelumeanlayer2()
testrelumeanlayer3()

