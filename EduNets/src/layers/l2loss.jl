
export L2Loss;


type L2Loss{T}<:AbstractLoss
  gX::Matrix{T};
end

function L2Loss(;T::DataType=Float64)
  return(L2Loss(zeros(T,0,0)));
end

function forward(loss::L2Loss,X::StridedMatrix,Y::StridedMatrix)
  f=sum((X-Y).^2)/size(Y,2);
  return(f/2);
end

function gradient!(loss::L2Loss,X::StridedMatrix,Y::StridedMatrix)
  if length(X)!=length(Y)
    error("L2Loss: X and Y has to have the same size")
  end
  if (size(loss.gX,2)<size(X,2)) && (size(loss.gX,1)<size(X,1))
    loss.gX=zeros(eltype(X),size(X))
  end

  l=size(Y,2);
  f=zero(eltype(X));
  @simd for i in 1:length(X)
    @inbounds loss.gX[i]=X[i]-Y[i];
    @inbounds f+=loss.gX[i]^2
    loss.gX[i]/=l
  end
  return(f/(2*l),loss.gX)
end