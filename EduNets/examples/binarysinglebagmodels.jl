function relurelumaxmodel(k::Tuple{Int,Int,Int})
  model=SingleBagModel(StackedBlocks(
        ReluLayer((k[1],k[2]),T=Float32),
        ReluLayer((k[2],k[3]),T=Float32),T=Float32),
      MaxPoolingLayer(k[3];T=Float32),
      LinearLayer((k[3],1);T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relurelumax_%d_%d",k[2],k[3])
  return(model,name)
end

function relumaxmodel(k::Tuple{Int,Int})
  model=SingleBagModel(ReluLayer((k[1],k[2]),T=Float32),
      MaxPoolingLayer(k[2];T=Float32),
      LinearLayer((k[2],1);T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relumax_%d",k[2])
  return(model,name)
end

function relumaxrelumodel(k::Tuple{Int,Int,Int})
  model=SingleBagModel(ReluLayer((k[1],k[2]),T=Float32),
      MaxPoolingLayer(k[2];T=Float32),
      StackedBlocks(
      ReluLayer((k[2],k[3]),T=Float32),
      LinearLayer((k[3],1),T=Float32),T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relumaxrelu_%d_%d",k[2],k[3])
  return(model,name)
end

function relurelumaxrelumodel(k::Tuple{Int,Int,Int,Int})
  model=SingleBagModel(StackedBlocks(
        ReluLayer((k[1],k[2]),T=Float32),
        ReluLayer((k[2],k[3]),T=Float32),T=Float32),
      MaxPoolingLayer(k[3];T=Float32),
      StackedBlocks(
      ReluLayer((k[3],k[4]),T=Float32),
      LinearLayer((k[4],1),T=Float32),T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relurelumaxrelu_%d_%d_%d",k[2],k[3],k[4])
  return(model,name)
end

function relurelumeanmodel(k::Tuple{Int,Int,Int})
  model=SingleBagModel(StackedBlocks(
        ReluLayer((k[1],k[2]),T=Float32),
        ReluLayer((k[2],k[3]),T=Float32),T=Float32),
      MeanPoolingLayer(k[3];T=Float32),
      LinearLayer((k[3],1);T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relurelumean_%d_%d",k[2],k[3])
  return(model,name)
end
function relumeanmodel(k::Tuple{Int,Int})
  model=SingleBagModel(
        ReluLayer((k[1],k[2]),T=Float32),
      MeanPoolingLayer(k[2];T=Float32),
      LinearLayer((k[2],1);T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relumean_%d",k[2])
  return(model,name)
end

function relumeanrelumodel(k::Tuple{Int,Int,Int})
  model=SingleBagModel(ReluLayer((k[1],k[2]),T=Float32),
      MeanPoolingLayer(k[2];T=Float32),
      StackedBlocks(
      ReluLayer((k[2],k[3]),T=Float32),
      LinearLayer((k[3],1),T=Float32),T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relumeanrelu_%d_%d",k[2],k[3])
  return(model,name)
end

function relurelumeanrelumodel(k::Tuple{Int,Int,Int,Int})
  model=SingleBagModel(StackedBlocks(
        ReluLayer((k[1],k[2]),T=Float32),
        ReluLayer((k[2],k[3]),T=Float32),T=Float32),
      MeanPoolingLayer(k[3];T=Float32),
      StackedBlocks(
      ReluLayer((k[3],k[4]),T=Float32),
      LinearLayer((k[4],1),T=Float32),T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relurelumeanrelu_%d_%d_%d",k[2],k[3],k[4])
  return(model,name)
end


function relurelulqmodel(k::Tuple{Int,Int,Int})
  model=SingleBagModel(StackedBlocks(
        ReluLayer((k[1],k[2]),T=Float32),
        ReluLayer((k[2],k[3]),T=Float32),T=Float32),
      LqPoolingLayer(k[3];T=Float32),
      LinearLayer((k[3],1);T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relurelulq_%d_%d",k[2],k[3])
  return(model,name)
end
function relulqmodel(k::Tuple{Int,Int})
  model=SingleBagModel(
        ReluLayer((k[1],k[2]),T=Float32),
      LqPoolingLayer(k[2];T=Float32),
      LinearLayer((k[2],1);T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relulq_%d",k[2])
  return(model,name)
end

function relulqrelumodel(k::Tuple{Int,Int,Int})
  model=SingleBagModel(ReluLayer((k[1],k[2]),T=Float32),
      LqPoolingLayer(k[2];T=Float32),
      StackedBlocks(
      ReluLayer((k[2],k[3]),T=Float32),
      LinearLayer((k[3],1),T=Float32),T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relulqrelu_%d_%d",k[2],k[3])
  return(model,name)
end

function relurelulqrelumodel(k::Tuple{Int,Int,Int,Int})
  model=SingleBagModel(StackedBlocks(
        ReluLayer((k[1],k[2]),T=Float32),
        ReluLayer((k[2],k[3]),T=Float32),T=Float32),
      LqPoolingLayer(k[3];T=Float32),
      StackedBlocks(
      ReluLayer((k[3],k[4]),T=Float32),
      LinearLayer((k[4],1),T=Float32),T=Float32),
      HingeLoss([1f0,1f0]),T=Float32);
  name=@sprintf("relurelulqrelu_%d_%d_%d",k[2],k[3],k[4])
  return(model,name)
end

function relugprelugp(k,n;T=Float32)
  model=DoubleBagModel(ReluLayer((k[1],k[2]);T=T),
      GaussPoolingLayer(k[2],n[1];T=T),
      ReluLayer((k[2],k[3]);T=T),
      GaussPoolingLayer(k[3],n[2];T=T),
      LinearLayer((k[3],1);T=T),
      HingeLoss([one(T),one(T)];T=T);T=T);

  modelname=@sprintf("relugprelugp_%d_%d_%d_%d_%d_%d",k[1],k[2],k[3],n[1],n[2])
  return(trnmodel,tstmodel,modelname)
end



