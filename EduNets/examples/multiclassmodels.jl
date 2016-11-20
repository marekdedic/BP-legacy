function lrelumaxlrelumax(k;T=Float32)
  trnmodel=DoubleBagModel(LeakyReluLayer((k[1],k[2]);T=T),
      MaxPoolingLayer(k[2];T=T),
      LeakyReluLayer((k[2],k[3]);T=T),
      MaxPoolingLayer(k[3];T=T),
      LinearLayer((k[3],k[4]);T=T),
      SoftmaxLoss(;T=T);T=T);

  tstmodel=DoubleBagModel(LeakyReluLayer((k[1],k[2]);T=T),
    MaxPoolingLayer(k[2];T=T),
    LeakyReluLayer((k[2],k[3]);T=T),
    MaxPoolingLayer(k[3];T=T),
    StackedBlocks(LinearLayer((k[3],k[4]);T=T),
    SoftmaxLayer(k[4];T=T);T=T),
    SoftmaxLoss(;T=T);T=T);

  modelname=@sprintf("lrelumaxlrelumax_%d_%d_%d_%d",k[1],k[2],k[3],k[4])
  return(trnmodel,tstmodel,modelname)
end

function lrelumeanlrelumean(k;T=Float32)
  trnmodel=DoubleBagModel(LeakyReluLayer((k[1],k[2]);T=T),
      MeanPoolingLayer(k[2];T=T),
      LeakyReluLayer((k[2],k[3]);T=T),
      MeanPoolingLayer(k[3];T=T),
      LinearLayer((k[3],k[4]);T=T),
      SoftmaxLoss(;T=T);T=T);

  tstmodel=DoubleBagModel(LeakyReluLayer((k[1],k[2]);T=T),
    MeanPoolingLayer(k[2];T=T),
    LeakyReluLayer((k[2],k[3]);T=T),
    MeanPoolingLayer(k[3];T=T),
    StackedBlocks(LinearLayer((k[3],k[4]);T=T),
    SoftmaxLayer(k[4];T=T);T=T),
    SoftmaxLoss(;T=T);T=T);

  modelname=@sprintf("lrelumeanlrelumean_%d_%d_%d_%d",k[1],k[2],k[3],k[4])
  return(trnmodel,tstmodel,modelname)
end


function relumaxrelumax(k;T=Float32)
  trnmodel=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
      MaxPoolingLayer(k[2];T=T),
      ReluLayer((k[2],k[3]);T=T),
      MaxPoolingLayer(k[3];T=T),
      LinearLayer((k[3],k[4]);T=T),
      SoftmaxLoss(;T=T);T=T);

  tstmodel=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
    MaxPoolingLayer(k[2];T=T),
    ReluLayer((k[2],k[3]);T=T),
    MaxPoolingLayer(k[3];T=T),
    StackedBlocks(LinearLayer((k[3],k[4]);T=T),
    SoftmaxLayer(k[4];T=T);T=T),
    SoftmaxLoss(;T=T);T=T);

  modelname=@sprintf("relumaxrelumax_%d_%d_%d_%d",k[1],k[2],k[3],k[4])
  return(trnmodel,tstmodel,modelname)
end

function relumeanrelumean(k;T=Float32)
  trnmodel=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
      MeanPoolingLayer(k[2];T=T),
      ReluLayer((k[2],k[3]);T=T),
      MeanPoolingLayer(k[3];T=T),
      LinearLayer((k[3],k[4]);T=T),
      SoftmaxLoss(;T=T);T=T);

  tstmodel=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
    MeanPoolingLayer(k[2];T=T),
    ReluLayer((k[2],k[3]);T=T),
    MeanPoolingLayer(k[3];T=T),
    StackedBlocks(LinearLayer((k[3],k[4]);T=T),
    SoftmaxLayer(k[4];T=T);T=T),
    SoftmaxLoss(;T=T);T=T);

  modelname=@sprintf("relumeanrelumean_%d_%d_%d_%d",k[1],k[2],k[3],k[4])
  return(trnmodel,tstmodel,modelname)
end

function relugprelugp(k::NTuple{4,Int},n::NTuple{2,Int};T=Float32)
  trnmodel=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
      GaussPoolingLayer(k[2],n[1];T=T),
      ReluLayer((k[2],k[3]);T=T),
      GaussPoolingLayer(k[3],n[2];T=T),
      LinearLayer((k[3],k[4]);T=T),
      SoftmaxLoss(;T=T);T=T);

  tstmodel=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
    GaussPoolingLayer(k[2],n[1];T=T),
    ReluLayer((k[2],k[3]);T=T),
    GaussPoolingLayer(k[3],n[2];T=T),
    StackedBlocks(LinearLayer((k[3],k[4]);T=T),
    SoftmaxLayer(k[4];T=T);T=T),
    SoftmaxLoss(;T=T);T=T);

  modelname=@sprintf("relumeanrelumean_%d_%d_%d_%d",k[1],k[2],k[3],k[4])
  return(trnmodel,tstmodel,modelname)
end

function getmodel(modeltype,k;T=Float32)
  if modeltype=="relumaxrelumax"
    return(relumaxrelumax(k,T=T))
  elseif modeltype=="relumeanrelumean"
    return(relumeanrelumean(k,T=T))
  end
end