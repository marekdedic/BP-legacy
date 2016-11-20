import Base.length;

export ReluMeanLayer,length,size,layervector,update!,forward,forwardrnd,backprops,backprop,gradient

type ReluMeanLayer{T<:AbstractFloat}<:AbstractMatMulLayer
    W::Matrix{T};
    B::Vector{T};

    O::Matrix{T}
    maxI::BitArray
    gX::Matrix{T}
end

function ReluMeanLayer(k::Tuple{Int,Int};T::DataType=Float64)
    return(ReluMeanLayer{T}(randn(k[1],k[2]),randn(k[2]),zeros(T,0,0),BitArray(0,0),zeros(T,0,0)));
end

function length(model::ReluMeanLayer)
    return(length(model.W)+length(model.B))
end

function size(model::ReluMeanLayer)
    return(size(model.W));
end

function size(model::ReluMeanLayer,k::Int)
    return(size(model.W,k::Int));
end

function model2vector(model::ReluMeanLayer)
  theta=zeros(eltype(model.W),length(model.W)+length(model.B));
  theta[1:length(model.W)]=model.W;
  theta[length(model.W)+1:end]=model.B;
  return(theta);
end

function model2vector!(model::ReluMeanLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(theta,offset,model.W,1,length(model.W));
    Base.LinAlg.copy!(theta,offset+length(model.W),model.B,1,length(model.B));
    return(offset+length(model));
end

function add!(model::ReluMeanLayer,theta::AbstractArray;offset::Int=1)
  @simd for i in 1:length(model.W)
    @inbounds theta[offset+i-1]+=model.W[i]
  end
  offset+=length(model.W)

  @simd for i in 1:length(model.B)
    @inbounds theta[offset+i-1]+=model.B[i]
  end    
  offset+=length(model.B)
  return(offset);
end


function update!(model::ReluMeanLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(model.W,1,theta,offset,length(model.W))
    Base.LinAlg.copy!(model.B,1,theta,length(model.W)+offset,length(model.B))
    return(offset+length(model))
end

function l1regularize!(model::ReluMeanLayer,g::ReluMeanLayer,lambda::AbstractFloat)
  f=eltype(model.W)(0.0);
  f+=lambda*sumabs(model.W)
  g.W+=lambda*sign(model.W)
  return(f)
end


""" reluMean{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::AbstractArray{AbstractArray{Int64,1},1})
    bags identifies bags within the X matrix. Each bag contains array of indexes into x. The indexes does not needs to be continuous, which is the 
    advantage, but it is less efficient than the version with continuous blocks,
    W,B are weight matrix parametrising the relu
    returns tuple (O,maxI) with O being the output of the relus on bags, and maxI being indexes of winners inside the relu.
"""
function forward{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags) 
    #iterate over bags and biases
    O=zeros(T,size(layer.W,2),length(bags))
    maxI=BitArray(size(layer.W,2),size(X,2))
    forward!(layer,X,bags,O,maxI);

    return(O,maxI)
end

function forward!{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags) 
    if size(layer.O,1)<size(layer.W,2) || size(layer.O,2)<length(bags)
        layer.O=zeros(T,size(layer.W,2),length(bags))
    end
    if size(layer.maxI,1)<size(layer.W,2) || size(layer.maxI,2)<size(X,2)
        layer.maxI=BitArray(size(layer.W,2),size(X,2))
    end
    fill!(layer.O,0.0);
    fill!(layer.maxI,false);
    forward!(layer,X,bags,layer.O,layer.maxI);
    O=view(layer.O,:,1:length(bags));
end

function forward!{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags,O::StridedMatrix{T}) 
    if size(O,1)<size(layer.W,2) || size(O,2)<length(bags)
        error("reluMean::forward size of output matrix O is not large enough ")
    end
    if size(layer.maxI,1)<size(layer.W,2) || size(layer.maxI,2)<size(X,2)
        layer.maxI=BitArray(size(layer.W,2),size(X,2))
    end
    fill!(O,0.0);
    fill!(layer.maxI,false);
    forward!(layer,X,bags,O,layer.maxI);
    return(O)
end
function forward!{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags,O::StridedMatrix{T},maxI::BitArray) 
    n=size(layer.W,2);
    for i in 1:length(bags) #iterate over bags
        for j in bags[i]  #iterate over subbags
            for k in 1:n #iterate over neurons
                oo=layer.B[k]
                @simd for l in 1:size(X,1)
                    @inbounds oo+=X[l,j]*layer.W[l,k]
                end
                if oo>0
                    @inbounds O[k,i]+=oo;
                    @inbounds maxI[k,j]=true;
                end
            end
        end
        O[:,i]/=length(bags[i])
    end
end

function backprop!{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},g::ReluMeanLayer{T};update=false) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if size(layer.gX,1)<size(X,1) || size(layer.gX,2)<size(X,2)
        layer.gX=zeros(T,size(X));
    end
        
    if !update
        fill!(g.W,0)
        fill!(g.B,0)
    end
    fill!(layer.gX,0);
    backprop!(layer,X,bags,gO,layer.maxI,layer.gX,g.W,g.B);
    gX=view(layer.gX,1:size(X,1),1:size(X,2));
    return(gX)
end



function backprop!{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},maxI::BitArray,gX::StridedMatrix{T},gW::Matrix{T},gB::Vector{T}) 
    #iterate over bags and biases
    n=size(layer.W,2);
    for i in 1:length(bags) #iterate over bags
        bagsize=length(bags[i])
        for j in bags[i]  #iterate over subbags
            for w in 1:n #iterate over neurons
                if maxI[w,j]
                    @simd for k in 1:size(layer.W,1)
                        @inbounds gX[k,j]+=gO[w,i]*layer.W[k,w]/bagsize;    
                        @inbounds gW[k,w]+=gO[w,i]*X[k,j]/bagsize; 
                    end
                    @inbounds gB[w]+=gO[w,i]/bagsize;
                end
            end
        end
    end
    return(gX)
end

function gradient!{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},g::ReluMeanLayer{T};update=false) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if !update
        fill!(g.W,0)
        fill!(g.B,0)
    end
    gradient!(layer,X,bags,gO,layer.maxI,g.W,g.B);
end



function gradient!{T<:AbstractFloat}(layer::ReluMeanLayer,X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},maxI::BitArray,gW::Matrix{T},gB::Vector{T}) 
    #iterate over bags and biases
    n=size(layer.W,2);
    for i in 1:length(bags) #iterate over bags
        bagsize=length(bags[i])
        for j in bags[i]  #iterate over subbags
            for w in 1:n #iterate over neurons
                if maxI[w,j]>0
                    @simd for k in 1:size(layer.W,1)
                        @inbounds gW[k,w]+=gO[w,i]*X[k,j]/bagsize; 
                    end
                    @inbounds gB[w]+=gO[w,i]/bagsize;
                end
            end
        end
    end
end