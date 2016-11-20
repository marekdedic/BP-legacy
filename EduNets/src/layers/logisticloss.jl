
export forward,gradient,LogisticLoss;


type LogisticLoss<:AbstractLoss
  gX::Matrix;   #preallocated space for returning gradient with respect to input
  n::Int;       #number of valid items in gX
end

function LogisticLoss()
  return(LogisticLoss(zeros(0,0),0));
end

function forward{T}(loss::LogisticLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  #proceed with calculating the loss function
  O=1+exp(-Y.*(X*w));
  return(mean(log(O)));
end

function gradient!{T}(loss::LogisticLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  f=T(0);
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=1+exp(-y*X[i])
    f+=log(o)
    loss.gX[i]=-y*(1-1/o)/length(Y)
  end
  f/=length(Y)
  return(f,view(loss.gX,:,1:length(Y)))
end

function checksize!(X::AbstractArray,loss::LogisticLoss)
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end