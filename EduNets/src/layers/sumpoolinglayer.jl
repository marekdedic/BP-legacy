import Base.length;
using Base.CartesianRange;

export SumPoolingLayer,length,size,model2vector,update!,forward,forwardrnd,backprops,backprop,gradient

"""Pooling layer from Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks"""
type SumPoolingLayer{T<:AbstractFloat}<:AbstractPooling
    gX::Array{T,2}
    O::Array{T,2}
end

"""function SumPoolingLayer{T}(k::Int;q::T=2.0)
    k --- number of input units (the number of output units is the same)
"""
function SumPoolingLayer(k::Int;T::DataType=Float64)
    return(SumPoolingLayer{T}(zeros(T,0,0),zeros(T,0,0)));
end

function length(layer::SumPoolingLayer)
    return(0)
end

function model2vector{T}(layer::SumPoolingLayer{T})
    return(zeros(T,0));
end

function model2vector!(layer::SumPoolingLayer,theta::Vector;offset::Int=1)
    return(offset+length(layer));
end

function update!(layer::SumPoolingLayer,theta::Vector;offset::Int=1)
    return(offset+length(layer));
end

function init!(layer::SumPoolingLayer,X::StridedMatrix)
end


function forward{T<:AbstractFloat}(layer::SumPoolingLayer,X::StridedMatrix{T},bags::Bags) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    O=zeros(T,size(X,1),length(bags))
    forward!(layer,X,bags,O);
    return(O);
end

function backprop!{T<:AbstractFloat}(layer::SumPoolingLayer{T},X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},g::SumPoolingLayer{T};update=false)
    backprop!(layer,X,bags,gO)
end

function forward!{T<:AbstractFloat}(layer::SumPoolingLayer,X::StridedMatrix{T},bags::Bags) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if size(layer.O,1)<size(X,1) || size(layer.O,2)<length(bags)
        layer.O=zeros(T,size(X,1),length(bags))
    end
    forward!(layer,X,bags,layer.O);
    return(view(layer.O,1:size(X,1),1:length(bags)))
end


function forward!{T<:AbstractFloat}(layer::SumPoolingLayer,X::StridedMatrix{T},bags::Bags,O::StridedMatrix{T}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    #iterate over bags and biases
    fill!(O,0.0);
    for i in 1:length(bags) #iterate over bags
        for j in bags[i]  #iterate over items (vectors) in bags
            @simd for k in 1:size(X,1)
                @inbounds O[k,i]+=X[k,j];
            end
        end
    end
end

function backprop{T<:AbstractFloat}(layer::SumPoolingLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    gX=zeros(T,size(X));
    backprop!(layer,X,bags,gO,gX) 
    return(gX)
end

function backprop!{T<:AbstractFloat}(layer::SumPoolingLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T})
    if size(layer.gX,1)<size(X,1) || size(layer.gX,2)<size(X,2)
        layer.gX=zeros(T,size(X));
    end
    backprop!(layer,X,bags,gO,layer.gX)
    return(view(layer.gX,1:size(X,1),1:size(X,2)));
end

function backprop!{T<:AbstractFloat}(layer::SumPoolingLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},gX::StridedMatrix{T}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    fill!(gX,0);
    for i in 1:length(bags) #iterate over bags
        for j in bags[i]  #iterate over subbags
            @simd for k in 1:size(X,1)
                @inbounds gX[k,j]+=gO[k,i];    
            end
        end
    end
    return(gX)
end

@inline function l1regularize!{T<:AbstractFloat}(model::SumPoolingLayer,g::SumPoolingLayer,lambda::T)
  return(T(0))
end

@inline function logl1regularize!{T<:AbstractFloat}(model::SumPoolingLayer,g::SumPoolingLayer,lambda::T,epsilon::T;dimension::Int=2)
  return(T(0))
end