import Base.length;
using Base.CartesianRange;
export ReluMaxLayer,length,size,model2vector,update!,forward,backprop,gradient

type ReluMaxLayer{T<:AbstractFloat}<:AbstractPooling
    W::Array{T,2};
    B::Array{T,1};
    gX::Matrix{T};
    O::Matrix{T};
    maxI::Matrix{Int};
end

function length(model::ReluMaxLayer)
    return(length(model.W)+length(model.B))
end

function size(model::ReluMaxLayer)
    return(size(model.W));
end

function size(model::ReluMaxLayer,k::Int)
    return(size(model.W,k::Int));
end

function model2vector(model::ReluMaxLayer)
  theta=zeros(eltype(model.W),length(model.W)+length(model.B));
  theta[1:length(model.W)]=model.W;
  theta[length(model.W)+1:end]=model.B;
  return(theta);
end

function model2vector!(model::ReluMaxLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(theta,offset,model.W,1,length(model.W));
    Base.LinAlg.copy!(theta,offset+length(model.W),model.B,1,length(model.B));
    return(offset+length(model));
end

function add!(model::ReluMaxLayer,theta::AbstractArray;offset::Int=1)
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


function update!(model::ReluMaxLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(model.W,1,theta,offset,length(model.W))
    Base.LinAlg.copy!(model.B,1,theta,length(model.W)+offset,length(model.B))
    return(offset+length(model))
end


function l1regularize!(model::ReluMaxLayer,g::ReluMaxLayer,lambda::AbstractFloat)
  f=eltype(model.W)(0.0);
  f+=lambda*sumabs(model.W)
  g.W+=lambda*sign(model.W)
  return(f)
end

""" Regularizes by a log of L1 norm of rows / columns specified by the dimension dimension( 2 --- rows, 1--- columns) """
function logl1regularize!(model::ReluMaxLayer,g::ReluMaxLayer,lambda::AbstractFloat,epsilon::AbstractFloat;dimension=2)
  f=eltype(model.W)(0.0);
  arglog=mean(abs(model.W),dimension)+epsilon;
  f+=lambda*(mean(log(arglog))-log(epsilon))
  g.W+=lambda*broadcast(*,sign(model.W),1./(size(model.W,dimension)*arglog));
  return(f)
end


function ReluMaxLayer{T}(W::Array{T,2},B::Array{T,1})
    return(ReluMaxLayer(W,B,zeros(T,0,0),zeros(T,0,0),zeros(Int,0,0)))
end

function ReluMaxLayer(d::Int,k::Int;T::DataType=Float64)
    return(ReluMaxLayer{T}(randn(d,k),randn(k),zeros(0,0),zeros(0,0),zeros(Int,0,0)));
end

function ReluMaxLayer(k::Tuple{Int,Int};T::DataType=Float64)
    return(ReluMaxLayer{T}(randn(k),randn(k[2]),zeros(0,0),zeros(0,0),zeros(Int,0,0)));
end

@inline function forward!{T<:AbstractFloat}(model::ReluMaxLayer,X::AbstractArray{T,2},bags::Bags) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if size(model.O,1)<size(model.W,2) || size(model.O,2)<length(bags)
        model.O=zeros(T,size(model.W,2),length(bags))
        model.maxI=zeros(Int,size(model.W,2),length(bags))
    end

    forward!(model,X,bags,model.O,model.maxI)
    O=view(model.O,1:size(model.W,2),1:length(bags))
    return(O)
end

@inline function forward!{T<:AbstractFloat}(model::ReluMaxLayer,X::AbstractArray{T,2}{T},bags::Bags,O::StridedMatrix{T}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if size(model.maxI,1)<size(model.W,2) || size(model.maxI,2)<length(bags)
        model.maxI=zeros(Int,size(model.W,2),length(bags))
    end
    if size(O,1)<size(model.W,2) || size(O,2)<length(bags)
        error("ReluMaxLayer:: size if the preallocated output is too small to fit the output")
    end

    forward!(model,X,bags,O,model.maxI)
    return(O)
end

function forward!{T<:AbstractFloat}(model::ReluMaxLayer,X::AbstractArray{T,2},bags::Bags,O::StridedMatrix{T},maxI::StridedMatrix{Int}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    fill!(O,0)
    fill!(maxI,0)
    for i in 1:length(bags) #iterate over bags
        for j in bags[i]  #iterate over instances
            for k in 1:size(model.W,2); #iterate over neurons
                oo=model.B[k]
                @simd for l in 1:size(X,1)
                    @inbounds oo+=X[l,j]*model.W[l,k]
                end
                if oo>O[k,i]
                    O[k,i]=oo;
                    maxI[k,i]=j;
                end
            end
        end
    end
end

@inline function backprop!{T<:AbstractFloat}(model::ReluMaxLayer{T},X::AbstractArray{T,2},bags::Array{Array{Int,1},1},gO::AbstractArray{T,2},g::ReluMaxLayer{T};update=false) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    return(backprop!(model,X,gO,g,update=update));
end

@inline function backprop!{T<:AbstractFloat}(model::ReluMaxLayer,X::AbstractArray{T,2},gO::AbstractArray{T,2},g::ReluMaxLayer{T};update=false) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if size(model.gX,1)<size(X,1) || size(model.gX,2)<size(X,2)
        model.gX=zeros(T,size(X));
    end

    if !update
        fill!(g.W,0)
        fill!(g.B,0)
    end
    backprop!(model,X,gO,model.gX,g.W,g.B,model.maxI)
    return(view(model.gX,1:size(X,1),1:size(X,2)))
end

function backprop!{T<:AbstractFloat}(model::ReluMaxLayer,X::AbstractArray{T,2},gO::AbstractArray{T,2},gX::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T},maxI::StridedMatrix{Int}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    #iterate over bags and biases
    fill!(gX,0);
    for I in CartesianRange(size(gO))
        if maxI[I]>0 && abs(gO[I])>0
            w=I[1];
            @simd for k in 1:size(model.W,1)
                @inbounds gX[k,maxI[I]]+=gO[I]*model.W[k,w];    
                @inbounds gW[k,w]+=gO[I]*X[k,maxI[I]]; 
            end
            gB[w]+=gO[I];
        end
    end
    return(gX)
end


@inline function gradient!{T}(model::ReluMaxLayer,X::AbstractArray{T,2}{T},gO::AbstractArray{T,2},g::ReluMaxLayer{T};update=false) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    if !update
        fill!(g.W,0)
        fill!(g.B,0)
    end
    gradient!(model,X,gO,g.W,g.B,model.maxI)
end

function gradient!{T}(model::ReluMaxLayer,X::AbstractArray{T,2}{T},gO::AbstractArray{T,2},gW::StridedMatrix{T},gB::Vector{T},maxI::StridedMatrix{Int}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    #iterate over bags and biases
    for I in CartesianRange(size(gO))
        w=I[1]
        if maxI[I]>0 && abs(gO[I])>0   
            @simd for k in 1:size(model.W,1)
                @inbounds gW[k,w]+=gO[I]*X[k,maxI[I]]; 
            end
            gB[w]+=gO[I];
        end
    end
end


#--------------------------------------------------------------------------------------------------------------------------------
#SPARSE Matrices support
#--------------------------------------------------------------------------------------------------------------------------------

""" forward{T<:AbstractFloat}((model::ReluMaxLayer,X::SparseMatrixCSC{T,Int64},bags::AbstractArray{AbstractArray{Int64,1},1})
    bags identifies bags within the X matrix. Each bag contains array of indexes into x. The indexes does not needs to be continuous, which is the 
    advantage, but it is less efficient than the version with continuous blocks,
    W,B are weight matrix parametrising the relu
    returns tuple (O,maxI) with O being the output of the relus on bags, and maxI being indexes of winners inside the relu.
"""
function forward{T<:AbstractFloat}(model::ReluMaxLayer,X::SparseMatrixCSC{T,Int64},bags::Bags) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    #iterate over bags and biases
    O=zeros(T,size(model.W,2),length(bags))
    maxI=zeros(Int64,size(model.W,2),length(bags))
    for i in 1:length(bags) #iterate over bags
        for j in bags[i]  #iterate over subbags
            xxindexes=X.colptr[j]:(X.colptr[j+1]-1);
            nnzrow=X.rowval[xxindexes];
            nnzval=X.nzval[xxindexes];
            for k in 1:size(model.W,2) #iterate over neurons
                oo=LinAlg.dot(nnzval,model.W[nnzrow,k])+model.B[k]
                if oo>O[k,i]
                    O[k,i]=oo;
                    maxI[k,i]=j;
                end
            end
        end
    end
    return(O,maxI)
end

function gradient{T<:AbstractFloat}(model::ReluMaxLayer,X::SparseMatrixCSC{T,Int64},gO::AbstractArray{T,2},maxI::AbstractArray{Int64}) # 0.008853 seconds (152.04 k allocations: 4.354 MB)
    #iterate over bags and biases
    gW=zeros(T,size(model.W))
    gB=zeros(T,size(model.B))
    @inbounds for i in 1:size(maxI,2)
        for w in 1:size(model.W,2)
            if maxI[w,i]>0 && abs(gO[w,i])>0   
                gB[w]+=gO[w,i];
                xxindexes=X.colptr[maxI[w,i]]:(X.colptr[maxI[w,i]+1]-1);
                gW[X.rowval[xxindexes],:]+=gO[w,i]*X.nzval[xxindexes]; # LinAlg.BLAS.blascopy!
            end
        end 
    end
    return(ReluMaxLayer(gW,gB))
end

