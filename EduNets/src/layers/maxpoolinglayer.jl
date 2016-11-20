using Base.CartesianRange;
export MaxPoolingLayer

"""Pooling layer from Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks"""
type MaxPoolingLayer{T<:AbstractFloat}<:AbstractPooling
    maxI::Array{Int,2}
    gX::Array{T,2}
    O::Array{T,2}
end

"""function MaxPoolingLayer{T}(k::Int;q::T=2.0)
    k --- number of input units (the number of output units is the same)
"""
function MaxPoolingLayer(k::Int;T::DataType=Float64)
    return(MaxPoolingLayer{T}(zeros(Int,0,0),zeros(T,0,0),zeros(T,0,0)));
end

function length(layer::MaxPoolingLayer)
    return(0)
end

function model2vector{T}(layer::MaxPoolingLayer{T})
    return(zeros(T,0));
end

function init!(layer::MaxPoolingLayer,X::StridedMatrix)
end


function model2vector!(layer::MaxPoolingLayer,theta::AbstractArray;offset::Int=1)
    return(offset+length(layer));
end

function update!(layer::MaxPoolingLayer,theta::AbstractArray;offset::Int=1)
    return(offset+length(layer));
end

function add!(layer::MaxPoolingLayer,theta::AbstractArray;offset::Int=1)
    return(offset+length(layer));
end


function forward{T<:AbstractFloat}(layer::MaxPoolingLayer,X::StridedMatrix{T},bags::Bags) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    O=zeros(T,size(X,1),length(bags))
    maxI=zeros(Int,size(X,1),length(bags))
    forward!(layer,X,bags,O,maxI);
    return((O,maxI));
end

function forward!{T<:AbstractFloat}(layer::MaxPoolingLayer,X::StridedMatrix{T},bags::Bags) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if size(layer.O,1)<size(X,1) || size(layer.O,2)<length(bags)
        layer.O=zeros(T,size(X,1),length(bags))
        layer.maxI=zeros(Int,size(X,1),length(bags))
    end
    forward!(layer,X,bags,layer.O,layer.maxI);
    return(view(layer.O,1:size(X,1),1:length(bags)))
end


function forward!{T<:AbstractFloat}(layer::MaxPoolingLayer,X::StridedMatrix{T},bags::Bags,O::StridedMatrix{T},maxI::StridedMatrix{Int}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    #iterate over bags and biases
    fill!(O,-Inf);
    for i in 1:length(bags) #iterate over bags
        for j in bags[i]  #iterate over items (vectors) in bags
            for k in 1:size(X,1)
                if X[k,j]>O[k,i]
                    O[k,i]=X[k,j];
                    maxI[k,i]=j;
                end
            end
        end
    end
end


function backprop{T<:AbstractFloat}(layer::MaxPoolingLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},maxI::StridedMatrix{Int}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    gX=zeros(T,size(X));
    backprop!(layer,X,bags,gO,gX,maxI) 
    return(gX)
end

function backprop!{T<:AbstractFloat}(layer::MaxPoolingLayer{T},X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},g::MaxPoolingLayer{T};update=false)
    backprop!(layer,X,bags,gO)
end

function backprop!{T<:AbstractFloat}(layer::MaxPoolingLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T})
    if size(layer.gX,1)<size(X,1) || size(layer.gX,2)<size(X,2)
        layer.gX=zeros(T,size(X));
    end
    backprop!(layer,X,bags,gO,layer.gX,layer.maxI)
    return(view(layer.gX,1:size(X,1),1:size(X,2)));
end

function backprop!{T<:AbstractFloat}(layer::MaxPoolingLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},gX::StridedMatrix{T},maxI::StridedMatrix{Int}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    fill!(gX,0);
    @inbounds for I in CartesianRange(size(gO))
        if maxI[I]>0 
            gX[I[1],maxI[I]]=gO[I];    
        end
    end
    return(gX)
end

@inline function l1regularize!{T<:AbstractFloat}(model::MaxPoolingLayer,g::MaxPoolingLayer,lambda::T)
  return(T(0))
end

@inline function logl1regularize!{T<:AbstractFloat}(model::MaxPoolingLayer,g::MaxPoolingLayer,lambda::T,epsilon::T;dimension::Int=2)
  return(T(0))
end