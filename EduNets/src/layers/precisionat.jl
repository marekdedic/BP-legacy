
export forward,gradient,PrecisionAtLoss;


"""
  This loss sets the threshold at q-recall of the positive distribution.
  
"""
type PrecisionAtLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
  q::T  ##quantile on the positive class
end

function PrecisionAtLoss(q::AbstractFloat;T::DataType=Float64)
  return(PrecisionAtLoss(zeros(T,0,0),T(1-q)));
end

function length(loss::PrecisionAtLoss)
  return(0)
end

function forward!{T}(loss::PrecisionAtLoss{T},X,Y)
  f=zero(T);
  th=quantile(X[Y.==2],loss.q) #find the threshold of the quantile
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=T(1.0-y*(X[i]-th))
    if o>0
      f+=o;
    end
  end
  f/=length(Y)
  return(f)
end

function gradient!{T}(loss::PrecisionAtLoss{T},X,Y)
  checksize!(X,loss);
  f=zero(T);
  fill!(loss.gX,zero(T));
  th=quantile(X[Y.==2],loss.q) #find the threshold of the quantile
  @inbounds @simd for i=1:length(Y)
    y=2*(Y[i]-1)-1
    o=T(1.0-y*(X[i]-th))
    if o>0
      f+=o;
      loss.gX[i]=-y/length(Y);
    end
  end
  f/=length(Y)
  return(f,view(loss.gX,:,1:length(Y)))
end

function checksize!{T}(X::AbstractArray{T},loss::PrecisionAtLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end