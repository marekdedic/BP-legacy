import Base.length;

export BatchNormLayer,forward,forward!,backprop,backprop!,gradient,gradient!

type BatchNormLayer{T<:AbstractFloat}
    gamma::Vector{T};
    beta::Vector{T};

    mn::Vector{T}
    sqvar::Vector{T}
    gX::Matrix{T}
    cntx::Matrix{T}
    O::Matrix{T}
    epsilon::T
end

function BatchNormLayer(k::Int;T::DataType=Float64,epsilon::AbstractFloat=1e-6)
  return(BatchNormLayer{T}(zeros(T,k),zeros(T,k),zeros(T,k),zeros(T,k),zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),T(epsilon)));
end

function forward!{T}(model::BatchNormLayer{T},X::AbstractArray{T})
  #check the size of the cache to store O
  if size(model.O,2)<size(X,2)
    model.O=zeros(T,size(X));
    model.cntx=zeros(eltype(X),size(X));
  end
  O=view(model.O,:,1:size(X,2))
  cntx=view(model.cntx,:,1:size(X,2));

  mean!(model.mn,X);
  broadcast!(-,cntx,X,model.mn);
  mean!(model.sqvar,cntx.^2);
  model.sqvar[:]=sqrt(model.sqvar+model.epsilon)
  broadcast!(/,O,cntx,model.sqvar);  

  return(O)
end

function backprop!{T}(model::BatchNormLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},g::BatchNormLayer{T};update=false)
  #check the size for storing the gradient of gX
  if size(model.gX,1)<size(X,1) || size(model.gX,2)<size(X,2)
    model.gX=zeros(eltype(X),size(X));
  end
  l=size(X,2)

  cntx=view(model.cntx,:,1:size(X,2));
  O=view(model.O,:,1:size(X,2));

  gvar=-sum(O.*gO,2)./(l*model.sqvar.^2);
  gmu=-sum(gO,2)./model.sqvar - gvar.*sum(cntx,2);

  gx=view(model.gX,:,1:size(X,2));
  for i in 1:size(X,2)
    for j in 1:size(X,1)
      gx[j,i]=gO[j,i]/model.sqvar[j]+cntx[j,i]*gvar[j]+ gmu[j]/l ;
    end
  end
  return(gx)
end