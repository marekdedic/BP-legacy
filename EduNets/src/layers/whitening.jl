import Base.length;

export WhiteningLayer,length,size,model2vector,update!,forward,backprops,backprop,gradient

type WhiteningLayer{T<:AbstractFloat}<:AbstractLayer
    W::Matrix{T};
    B::Vector{T};
    gX::Matrix{T}
    O::Matrix{T}
end

function WhiteningLayer{T}(W::Matrix{T},B::Vector{T})
  @assert length(B)==1 "WhiteningLayer:: When W is Array, B has to be an array of length 1"
  return(WhiteningLayer{T}(W,B,zeros(T,0,0),zeros(T,0,0)))
end

function WhiteningLayer(d::Int;T::DataType=Float64)
  return(WhiteningLayer{T}(randn(d,1),randn(1),zeros(T,0,0),zeros(T,0,0)));
end

function WhiteningLayer(d::Int,k::Int;T::DataType=Float64)
  return(WhiteningLayer{T}(randn(d,k),randn(k),zeros(T,0,0),zeros(T,0,0)));
end

function length(model::WhiteningLayer)
  return(length(model.W)+length(model.B))
end

function model2vector(model::WhiteningLayer)
  theta=zeros(eltype(model.W),length(model.W)+length(model.B));
  theta[1:length(model.W)]=model.W;
  theta[length(model.W)+1:end]=model.B;
  return(theta);
end

function model2vector!(model::WhiteningLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(theta,offset,model.W,1,length(model.W));
    Base.LinAlg.copy!(theta,offset+length(model.W),model.B,1,length(model.B));
    return(length(model));
end

function add!(model::WhiteningLayer,theta::AbstractArray;offset::Int=1)
  for i in 1:length(model.W)
    theta[offset+i-1]+=model.W[i]
  end
  offset+=length(model.W)

  for i in 1:length(model.B)
    theta[offset+i-1]+=model.B[i]
  end    
  offset+=length(model.B)
  return(offset);
end


function update!(model::WhiteningLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(model.W,1,theta,offset,length(model.W))
    Base.LinAlg.copy!(model.B,1,theta,length(model.W)+offset,length(model.B))
    return(length(model))
end



"""function absorb{T}(model::WhiteningLayer{T},follower::AbstractLayer)
returns follower's model with absorbed whitening layer.
The parameters are returned as a vector. """
function absorb{T}(model::WhiteningLayer{T},follower::AbstractLayer)
  W=model.W*follower.W;
  B=follower.B+follower.W'*model.B;
  theta=zeros(eltype(W),length(W)+length(B));
  theta[1:length(W)]=W;
  theta[length(W)+1:end]=B;
  return(theta);
end


function update!{T}(model::WhiteningLayer{T},theta::AbstractArray{T};offset::Int=1)
    Base.LinAlg.copy!(model.W,1,theta,offset,length(model.W))
    Base.LinAlg.copy!(model.B,1,theta,length(model.W)+offset,length(model.B))
end

function rewhite!{T}(model::WhiteningLayer{T},follower::AbstractLayer,sigma::AbstractArray{T,2},mu::AbstractArray{T,2};epsilon::T=1e-8)
  #do the notation as is in the paper "Natural Neural Networks, 2015"
  #calculate the canonical representation
  b=follower.B+follower.W'*model.B;
  W=model.W*follower.W;

  #updates the centering part
  (lambdas,U)=eig(sigma)
  lambdas+=epsilon
  model.W=U*diagm(1./sqrt(lambdas));
  model.B=-model.W'*squeeze(mu,2);

  #update the following layer
  follower.W=model.W\W # Sprechen Sie matlab???
  follower.B=b-follower.W'*model.B;
end

function forward{T}(X::AbstractArray{T,2},model::WhiteningLayer{T})
  O=model.W'*X;
  O=broadcast!(+,O,O,model.B);
  return(O)
end


function backprop{T}(model::WhiteningLayer{T},gO::AbstractArray{T,2})
  gX=model.W*gO';
  return(gX)
end


""" function forward!{T}(X::StridedMatrix{T},model::WhiteningLayer{T})
this version relies on BLAS library"""
function forward!{T}(X::StridedMatrix{T},model::WhiteningLayer{T})
  checksize!(X,model);
  O=view(model.O,1:size(model.W,2),1:size(X,2))
  Base.LinAlg.BLAS.gemm!('T','N',one(T),model.W,X,T(0),O)
  O=broadcast!(+,O,O,model.B);
  return(O)
end

function backprop!{T}(model::WhiteningLayer{T},gO::StridedMatrix{T})
  gX=view(model.gX,1:size(model.W,1),1:size(gO,2));
  Base.LinAlg.BLAS.gemm!('N','N',one(T),model.W,gO,T(0),gX);
  return(gX)
end

""" function forward!{T}(X::StridedMatrix{T},model::WhiteningLayer{T})
this version uses triple loop"""
function forward!{T}(X::AbstractArray{T,2},model::WhiteningLayer{T})
  checksize!(X,model);
  O=view(model.O,1:size(model.W,2),1:size(X,2));
  gemmTN!(model.W,X,model.B,O)
  return(O)
end

function backprop!{T}(model::WhiteningLayer{T},gO::AbstractArray{T})
  gX=view(model.gX,1:size(model.W,1),1:size(gO,2));
  gemmNN!(model.W,gO,gX);
  return()
end

function checksize!{T}(X::AbstractArray{T,2},model::WhiteningLayer{T})
  if size(model.O,1)<size(model.W,2) || size(model.O,2)<size(X,2)
    model.O=zeros(eltype(X),size(model.W,2),size(X,2));
  end

  if size(model.gX,1)<size(X,1) || size(model.gX,2)<size(X,2)
    model.gX=zeros(eltype(X),size(X));
  end
end