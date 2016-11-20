
export forward,gradient,MultiHingeLoss;


type MultiHingeLoss{T}<:AbstractLoss
  cost::Matrix{T};   #Cost matrix, where each row corresponds to correct class and column corresponds to predicted
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function length(loss::MultiHingeLoss)
  return(length(loss.cost))
end

function MultiHingeLoss(k::Int;T::Type=Float64)
  return(MultiHingeLoss(ones(T,k,k),zeros(0,0)));
end

function MultiHingeLoss{T}(cost::Matrix{T})
  return(MultiHingeLoss(cost,zeros(0,0)));
end

function forward{T}(loss::MultiHingeLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  f=T(0);
  for i in 1:size(X,2)
    @simd for j in 1:size(X,1)
      @inbounds y=2*(Y[i]==j)-1
      @inbounds o=1-y*X[j,i]
      if o>0
        f+=o;
      end
    end
  end
  return(f/length(Y));
end

function gradient{T}(loss::MultiHingeLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  gX=zeros(T,size(X));
  return(gradient!(loss,X,Y,gX));
end

function gradient!{T}(loss::MultiHingeLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
  gX=view(loss.gX,1:size(X,1),1:size(X,2));
  return(gradient!(loss,X,Y,gX));
end
  
function gradient!{T}(loss::MultiHingeLoss,X::AbstractArray{T,2},Y::AbstractArray{Int},gX::AbstractArray{T,2})
  f=T(0);
  fill!(gX,0.0);
  for i in 1:size(X,2)
    @simd for j in 1:size(X,1)
      @inbounds y=2*(Y[i]==j)-1
      @inbounds o=1-y*X[j,i]
      if o>0
        @inbounds f+=loss.cost[j]*o;
        @inbounds loss.gX[j,i]=-loss.cost[Y[i],j]*y/length(Y);
      end
    end
  end
  return(f/length(Y),gX);
end