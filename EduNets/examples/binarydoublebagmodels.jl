
function relumaxrelumax(k,w;T=Float32)
  model=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
      MaxPoolingLayer(k[2];T=T),
      ReluLayer((k[2],k[3]);T=T),
      MaxPoolingLayer(k[3];T=T),
      LinearLayer((k[3],1);T=T),
      HingeLoss([1-w,w]),T=T);
  modelname=@sprintf("relumaxrelumax_%d_%d_%d_%g",k[1],k[2],k[3],w)
  return(model,modelname)
end


function relumeanrelumean(k,w;T=Float32)
  model=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
      MeanPoolingLayer(k[2];T=T),
      ReluLayer((k[2],k[3]);T=T),
      MeanPoolingLayer(k[3];T=T),
      LinearLayer((k[3],1);T=T),
      HingeLoss([1-w,w]),T=T);
  modelname=@sprintf("relumeanrelumean_%d_%d_%d_%g",k[1],k[2],k[3],w)
  return(model,modelname)
end


function getmodel(modeltype,k,w;T=Float32)
  if modeltype=="relumaxrelumax"
    return(relumaxrelumax(k,w,T=T))
  elseif modeltype=="relumeanrelumean"
    return(relumeanrelumean(k,w,T=T))
  end
end