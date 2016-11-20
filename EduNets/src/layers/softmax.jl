
import Base.length;

export SoftmaxLayer,length,size,model2vector,update!,forward,backprop,gradient

type SoftmaxLayer{T<:AbstractFloat}<:AbstractLayer
    gX::Array{T,2};
    O::Array{T,2};
end

function SoftmaxLayer(k::Int;T::DataType=Float64)
    return(SoftmaxLayer(zeros(T,0,0),zeros(T,0,0)));
end

function length(layer::SoftmaxLayer)
    return(0)
end

function model2vector{T}(layer::SoftmaxLayer{T})
  return(zeros(T,0));
end

function model2vector!(layer::SoftmaxLayer,theta::AbstractArray;offset::Int=1)
    return(offset);
end

function update!(layer::SoftmaxLayer,theta::AbstractArray;offset::Int=1)
    return(offset)
end

function add!(layer::SoftmaxLayer,theta::AbstractArray;offset::Int=1)
    return(offset)
end

function forward!{T<:AbstractFloat}(layer::SoftmaxLayer,X::AbstractMatrix{T})
    if size(layer.O,1)<size(X,1) || size(layer.O,2)<size(X,2)
        layer.O=zeros(T,size(X));
    end

    O=view(layer.O,:,1:size(X,2));
    forward!(layer,X,O);
    return(O)
end

function forward!{T}(layer::SoftmaxLayer{T},X::StridedMatrix{T},O::StridedMatrix{T})
  for i in 1:size(X,2)
    #find the maximum for the stability
    maxval=typemin(T)
    @simd for j in 1:size(X,1)
      @inbounds maxval=(X[j,i]>maxval)?X[j,i]:maxval;
    end
    the_sum=0.0;
    @simd for j in 1:size(X,1)
      @inbounds O[j,i] = exp(X[j,i]-maxval)
      the_sum+=O[j,i]
    end
    @simd for j in 1:size(X,1)
      @inbounds O[j,i]/= the_sum
    end
  end
end

function backprop!{T}(layer::SoftmaxLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},g::SoftmaxLayer{T};update=false)
    if size(layer.gX,1)<size(X,1) || size(layer.gX,2)<size(X,2)
        layer.gX=zeros(T,size(X));
    end
    backprop!(layer,X,gO,layer.gX);
    gX=view(layer.gX,:,1:size(X,2));
    return(gX)
end

function backprop!{T}(layer::SoftmaxLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},gX::StridedMatrix{T})
  for i in 1:size(X,2)
    dot_prod=0.0
    for j in 1:size(X,1)
      @inbounds dot_prod += gO[j,i] * layer.O[j,i]
    end

    for j in 1:size(X,1)
      @inbounds gX[j,i] = (gO[j,i]-dot_prod)*layer.O[j,i]
    end
  end
end