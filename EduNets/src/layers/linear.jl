import Base.length;

export LinearLayer,forward,forward!,backprop,backprop!,gradient,gradient!

type LinearLayer{T<:AbstractFloat}<:AbstractMatMulLayer
    W::Matrix{T};
    B::Vector{T};
    gX::Matrix{T}
    O::Matrix{T}
end

function LinearLayer{T}(W::Matrix{T},B::Vector{T})
  return(LinearLayer{T}(W,B,zeros(T,0,0),zeros(T,0,0)))
end

function LinearLayer(k1::Int;T::DataType=Float64)
  return(LinearLayer{T}(randn(k1,1),randn(1),zeros(T,0,0),zeros(T,0,0)));
end

function LinearLayer(k::Tuple{Int,Int};T::DataType=Float64)
  return(LinearLayer{T}(randn(k[1],k[2]),randn(k[2]),zeros(T,0,0),zeros(T,0,0)));
end

function forward{T}(model::LinearLayer{T},X::StridedMatrix{T})
  O=zeros(T,size(model.W,2),size(X,2));
  forward!(model,X,O)
  return(O)
end

function forward!{T}(model::LinearLayer{T},X::AbstractArray{T})
  #check the size of the cache to store O
  if size(model.O,1)<size(model.W,2) || size(model.O,2)<size(X,2)
    model.O=zeros(eltype(X),size(model.W,2),size(X,2));
  end

  O=view(model.O,1:size(model.W,2),1:size(X,2))
  forward!(model,X,O);
  return(O)
end

function forward!{T}(model::LinearLayer{T},X::StridedMatrix{T},O::StridedMatrix{T})
  Base.LinAlg.BLAS.gemm!('T','N',one(T),model.W,X,T(0),O)
 # for i in 1:size(X,2)
 #      @simd for j in 1:size(X,1)
 #          @inbounds O[j,i]+=model.B[j];
 #      end
 #  end
  broadcast!(+,O,O,model.B);
  return(O)
end

function backprop(model::LinearLayer,X::StridedMatrix,gO::StridedMatrix)
  gX=zeros(eltype(X),size(X))
  gB=zeros(eltype(model.B),size(model.B));
  gW=zeros(eltype(model.W),size(model.W));
  backprop!(model,X,gO,gX,gW,gB);
  return(LinearLayer(gW,gB),gX)
end

function backprop!{T}(model::LinearLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},g::LinearLayer{T};update=false)
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

function backprop!{T}(model::LinearLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},gX::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T})
  Base.LinAlg.BLAS.gemm!('N','N',one(T),model.W,gO,zero(T),gX);
  Base.LinAlg.BLAS.gemm!('N','T',one(T),X,gO,one(T),gW);
  for I in CartesianRange(size(gO))
    gB[I[1]]+=gO[I]
  end
  return(gX)
end


function gradient{T}(model::LinearLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T})
  gB=zeros(eltype(model.B),size(model.B));
  gW=zeros(eltype(model.W),size(model.W));
  gradient!(model,X,gO,g.W,g.B);
  return(LinearLayer(gW,gB))
end

function gradient!{T}(model::LinearLayer{T},X::StridedMatrix{T},gO::StridedMatrix,g::LinearLayer{T};update=false)
  if !update
    fill!(g.B,0);
    fill!(g.W,0);
  end
  gradient!(model,X,gO,g.W,g.B);
end

function gradient!{T}(model::LinearLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T})
  Base.LinAlg.BLAS.gemm!('N','T',one(T),X,gO,one(T),gW);
  for I in CartesianRange(size(gO))
    gB[I[1]]+=gO[I]
  end
end


function forward!{T}(model::LinearLayer{T},X::SparseMatrixCSC{T},O::StridedMatrix{T})
  for c in 1:X.n
    for w in 1:size(model.W,2)
      @inbounds O[w,c]=model.B[w];
      @simd for k in X.colptr[c]:(X.colptr[c+1]-1)
        @inbounds O[w,c]+=X.nzval[k]*model.W[X.rowval[k],w];
      end
    end 
  end
end

function gradient!{T}(model::LinearLayer{T},X::SparseMatrixCSC{T},gO::StridedMatrix{T},gW::StridedMatrix{T},gB::Vector{T})
  for c in 1:X.n
    for w in 1:size(model.W,2)
      @inbounds gB[w]+=gO[w,c];
      @simd for k in X.colptr[c]:(X.colptr[c+1]-1)
        @inbounds gW[X.rowval[k],w]+=X.nzval[k]*gO[w,c];
      end
    end
  end
end