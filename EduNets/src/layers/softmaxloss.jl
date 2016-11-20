
export forward,gradient,SoftmaxLoss;


type SoftmaxLoss{T}<:AbstractLoss
  w::Vector{T}
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function length(loss::SoftmaxLoss)
  return(0)
end

function SoftmaxLoss(k::Int;T::Type=Float64)
  return(SoftmaxLoss(ones(T,k),zeros(T,0,0)));
end

function SoftmaxLoss(w::Vector;T::Type=Float64)
  return(SoftmaxLoss(map(T,w),zeros(T,0,0)));
end

function forward{T}(loss::SoftmaxLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  sigmas=zeros(T,size(X,1))
  f=zero(T);
  for i in 1:size(X,2)
    #find the maximum for the stability
    maxval=typemin(T)
    @simd for j in 1:size(X,1)
      @inbounds maxval=(X[j,i]>maxval)?X[j,i]:maxval;
    end

    #calculate exponentials and their sums
    the_sum=T(0);
    @simd for j in 1:size(X,1)
      @inbounds sigmas[j] = exp(X[j,i]-maxval)
      the_sum+=sigmas[j]
    end

    #normalize
    @simd for j in 1:size(X,1)
      @inbounds sigmas[j]/= the_sum
    end

    #calculate the loss, ensure that it is non-zero
    if sigmas[Y[i]]>0
      f-=loss.w[Y[i]]*log(sigmas[Y[i]])
    else
      f-=loss.w[Y[i]]*log(1e-99);
    end
  end
  f/=size(X,1)
  return(f);
end

function gradient{T}(loss::SoftmaxLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  gX=zeros(T,size(X));
  return(gradient!(loss,X,Y,gX));
end

function gradient!{T}(loss::SoftmaxLoss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
  gX=view(loss.gX,1:size(X,1),1:size(X,2));
  return(gradient!(loss,X,Y,gX));
end
  
function gradient!{T}(loss::SoftmaxLoss,X::AbstractArray{T,2},Y::AbstractArray{Int},gX::AbstractArray{T,2})
  f=zero(T);
  for i in 1:size(X,2)
    #find the maximum for the stability
    maxval=typemin(T)
    @simd for j in 1:size(X,1)
      @inbounds maxval=(X[j,i]>maxval)?X[j,i]:maxval;
    end

    #calculate exponentials and their sums
    the_sum=zero(T);
    @simd for j in 1:size(X,1)
      @inbounds gX[j,i] = exp(X[j,i]-maxval)
      the_sum+=gX[j,i]
    end

    if gX[Y[i],i]>0
      f-=loss.w[Y[i]]*log(gX[Y[i],i]/the_sum)
    else
      f-=loss.w[Y[i]]*log(1e-99);
    end


    #normalize
    @simd for j in 1:size(X,1)
      @inbounds gX[j,i]*=loss.w[Y[i]]/the_sum
    end
    gX[Y[i],i]-=loss.w[Y[i]]
  end

  f/=size(X,1)
  gX/=size(X,1)
  return(f,gX);
end