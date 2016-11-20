import Base.length;
export ReluLayer,forward,forward!,backprop,backprop!,gradient,gradient!
# using ParallelAccelerator

type ReluLayer{T<:AbstractFloat}<:AbstractMatMulLayer
    W::Matrix{T};
    B::Vector{T};
    gX::Matrix{T}
    O::Matrix{T}
end

function ReluLayer(k::Tuple{Int,Int};T::DataType=Float64)
  return(ReluLayer{T}(randn(k[1],k[2]),randn(k[2]),zeros(T,0,0),zeros(T,0,0)));
end

function forward{T}(model::ReluLayer{T},X::StridedMatrix{T})
  O=zeros(T,size(model.W,2),size(X,2));
  forward!(model,X,O)
  return(O)
end

function forward!{T}(model::ReluLayer{T},X::AbstractArray{T})
  #check the size of the cache to store O
  if size(model.O,1)<size(model.W,2) || size(model.O,2)<size(X,2)
    model.O=zeros(T,size(model.W,2),size(X,2));
  end

  O=view(model.O,1:size(model.W,2),1:size(X,2))

  # @acc begin
    Base.LinAlg.BLAS.gemm!('T','N',one(T),model.W,X,zero(T),O)
    broadcast!(+,O,O,model.B);
    O=max(O,zero(T))
  # end
  return(O)
end


function backprop(model::ReluLayer,X::StridedMatrix,gO::StridedMatrix)
  gX=zeros(eltype(X),size(X))
  gB=zeros(eltype(model.B),size(model.B));
  gW=zeros(eltype(model.W),size(model.W));
  backprop!(model,X,gO,gX,gW,gB);
  return(ReluLayer(gW,gB),gX)
end

function backprop!{T}(model::ReluLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},g::ReluLayer{T};update=false)
  #check the size for storing the gradient of gX
  if size(model.gX,1)<size(X,1) || size(model.gX,2)<size(X,2)
    model.gX=zeros(eltype(X),size(X));
  end

  if !update
    fill!(g.B,0);
    fill!(g.W,0);
  end
  gX=view(model.gX,1:size(X,1),1:size(X,2));
  backprop!(model,X,gO,gX,g.W,g.B);
  return(gX)
end

function backprop!{T}(model::ReluLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},gX::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T})
  # @acc begin
    O=view(model.O,:,1:size(X,2))
    gO[O.<=zero(T)]=zero(T)
    Base.LinAlg.BLAS.gemm!('N','N',one(T),model.W,gO,zero(T),gX);
    Base.LinAlg.BLAS.gemm!('N','T',one(T),X,gO,one(T),gW);
    gB[:]=sum(gO,2)
  # end
  return(gX)
end


function gradient{T}(model::ReluLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T})
  gB=zeros(eltype(model.B),size(model.B));
  gW=zeros(eltype(model.W),size(model.W));
  gradient!(model,X,gO,g.W,g.B);
  return(ReluLayer(gW,gB))
end

function gradient!{T}(model::ReluLayer{T},X::StridedMatrix{T},gO::StridedMatrix,g::ReluLayer{T};update=false)
  if !update
    fill!(g.B,0);
    fill!(g.W,0);
  end
  gradient!(model,X,gO,g.W,g.B);
end

function gradient!{T}(model::ReluLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T})
  # @acc begin
    O=view(model.O,:,1:size(X,2))
    gO[O.<=zero(T)]=zero(T)
    Base.LinAlg.BLAS.gemm!('N','T',one(T),X,gO,one(T),gW);
    gB[:]=sum(gO,2)
  # end
end