
export forward,gradient,EqualHingeLoss;


type EqualHingeLoss<:AbstractLoss
  gX::Matrix;   #preallocated space for returning gradient with respect to input
  n::Int;       #number of valid items in gX
end

function EqualHingeLoss()
  return(EqualHingeLoss(zeros(0,0),0));
end

function length(loss::EqualHingeLoss)
  return(0)
end

function forward{T}(loss::EqualHingeLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  checksize!(X,loss);
  @assert size(X,1)==1 "input to hingeLossG has to be column matrix with one column"
  Y=2*(Y-1)-1;

  #proceed with calculating the loss function
  O=1-Y.*X';
  mask=O.<0;
  O[mask]=0;

  negmask=Y.<0;
  posmask=Y.>0;
  O[negmask]=O[negmask]*sum(posmask)/length(Y)
  O[posmask]=O[posmask]*sum(negmask)/length(Y)
  return(mean(O));
end

function gradient{T}(loss::EqualHingeLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  @assert size(X,1)==1 "input to hingeLossG has to be column matrix with one column"
  W=[sum(Y.==2)/length(Y);sum(Y.==1)/length(Y)]

  g=zeros(eltype(X),size(X))
  f=T(0);
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=1-y*X[i]
    if o>0
      f+=W[Y[i]]*o;
      g[i]=-W[Y[i]]*y/length(Y);
    end
  end
  f/=length(Y)
  return(f,g)
end

function gradient!{T}(loss::EqualHingeLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  checksize!(X,loss);
  W=[sum(Y.==2)/length(Y);sum(Y.==1)/length(Y)]

  f=T(0);
  fill!(loss.gX,0);
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=1-y*X[i]
    if o>0
      f+=W[Y[i]]*o;
      loss.gX[i]=-W[Y[i]]*y/length(Y);
    end
  end
  f/=length(Y)
  return(f,view(loss.gX,:,1:length(Y)))
end

function checksize!(X::AbstractArray,loss::EqualHingeLoss)
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end