import Base.length;
export EluLayer,forward,forward!,backprop,backprop!,gradient,gradient!
# using ParallelAccelerator

type EluLayer{T<:AbstractFloat}<:AbstractMatMulLayer
    #learnable parameters
    W::Matrix{T};
    B::Vector{T};

    #state of the network network
    gX::Matrix{T}
    O::Matrix{T}
    xf::Matrix{T}
    gO::Matrix{T}
    alpha::T
end

function EluLayer(k::Tuple{Int,Int};T::DataType=Float64,alpha::AbstractFloat=1.0)

  return(EluLayer{T}(randn(k[1],k[2]),randn(k[2]),zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),T(alpha)));
end

function forward{T}(model::EluLayer{T},X::StridedMatrix{T})
  O=zeros(T,size(model.W,2),size(X,2));
  forward!(model,X,O)
  return(O)
end

function forward!{T}(model::EluLayer{T},X::AbstractArray{T})
  #check the size of the cache to store O
  if size(model.O,2)<size(X,2)
    model.O=zeros(T,size(model.W,2),size(X,2));
    model.gO=zeros(T,size(model.W,2),size(X,2));
    model.xf=zeros(T,size(model.W,2),size(X,2));
  end

  xf=view(model.xf,:,1:size(X,2))
  Base.LinAlg.BLAS.gemm!('T','N',one(T),model.W,X,zero(T),xf)
  broadcast!(+,xf,xf,model.B);

  #apply the transfer function
  O=view(model.O,:,1:size(X,2))
  for i in 1:length(xf)
      O[i]=(xf[i]>=0)?xf[i]:model.alpha*(exp(xf[i])-1)
  end
  return(O)
end

function backprop!{T}(model::EluLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},g::EluLayer{T};update=false)
  #check the size for storing the gradient of gX
  if size(model.gX,2)<size(X,2)
    model.gX=zeros(T,size(X));
  end

  if !update
    fill!(g.B,0);
    fill!(g.W,0);
  end

  gX=view(model.gX,:,1:size(X,2));
  backprop!(model,X,gO,gX,g.W,g.B);
  return(gX)
end

function backprop!{T}(model::EluLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},gX::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T})
  #add the effect of transfer function
  ggO=view(model.gO,:,1:size(gO,2));
  O=view(model.O,:,1:size(X,2));
  for i in 1:length(O)
    ggO[i]=(O[i]>=0)?gO[i]:(model.alpha+O[i])gO[i];
  end

  #calculate the gradients
  Base.LinAlg.BLAS.gemm!('N','N',one(T),model.W,ggO,zero(T),gX);
  Base.LinAlg.BLAS.gemm!('N','T',one(T),X,ggO,one(T),gW);
  gB[:]=sum(ggO,2)
  return(gX)
end

function gradient!{T}(model::EluLayer{T},X::StridedMatrix{T},gO::StridedMatrix,g::EluLayer{T};update=false)
  if !update
    fill!(g.B,0);
    fill!(g.W,0);
  end
  gradient!(model,X,gO,g.W,g.B);
end

function gradient!{T}(model::EluLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T})
  #add the effect of transfer function
  ggO=view(model.gO,:,1:size(gO,2));
  O=view(model.O,:,1:size(X,2));
  for i in 1:length(O)
    ggO[i]=(O[i]>=0)?gO[i]:(model.alpha+O[i])gO[i];
  end

  #calculate the gradients
  Base.LinAlg.BLAS.gemm!('N','T',one(T),X,ggO,one(T),gW);
  gB[:]=sum(ggO,2)
end