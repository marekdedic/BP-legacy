
export forward,gradient,HingeLoss;


type HingeLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
  w::Vector{T}
end

function HingeLoss(;T::DataType=Float64)
  return(HingeLoss(zeros(T,0,0),ones(T,2)));
end

function HingeLoss{T}(w::Vector{T})
  return(HingeLoss(zeros(T,0,0),w));
end

function length(loss::HingeLoss)
  return(0)
end

function forward{T}(loss::HingeLoss{T},X::Matrix{T},Y::AbstractArray{Int})
  checksize!(X,loss);

  #proceed with calculating the loss function
  f=T(0);
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=T(1.0-y*X[i])
    if o>0
      f+=loss.w[Y[i]]*o;
    end
  end
  return(f/length(Y));
end

function gradient{T}(loss::HingeLoss{T},X::Matrix{T},Y)
  g=zeros(eltype(X),size(X))
  f=zero(T)
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=T(1.0-y*X[i])
    if o>0
      f+=loss.w[Y[i]]*o;
      g[i]=-loss.w[Y[i]]*y/length(Y);
    end
  end
  f/=length(Y)
  return(f,g)
end

function gradient!{T}(loss::HingeLoss{T},X,Y)
  checksize!(X,loss);
  f=zero(T);
  fill!(loss.gX,zero(T));
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=T(1.0-y*X[i])
    if o>0
      f+=loss.w[Y[i]]*o;
      loss.gX[i]=-loss.w[Y[i]]*y/length(Y);
    end
  end
  f/=length(Y)
  return(f,view(loss.gX,:,1:length(Y)))
end

"""This is a version for importance sampling. Ensure that w is properly scaled, it should sum up to one"""
function gradient!{T}(loss::HingeLoss{T},X::Matrix{T},Y,w::AbstractArray{T,1})
  checksize!(X,loss);
  f=zeros(T,size(X,2));
  fill!(loss.gX,zero(T));
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=T(1.0-y*X[i])
    if o>0
      f[i]=loss.w[Y[i]]*o;
      loss.gX[i]=-loss.w[Y[i]]*y/w[i];
    end
  end
  return(f,view(loss.gX,:,1:length(Y)))
end

function checksize!{T}(X::AbstractArray{T},loss::HingeLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end