
export forward,gradient,Fp50LossHinge;


type Fp50LossHinge{T}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function Fp50LossHinge(;T::DataType=Float64)
  return(Fp50LossHinge(zeros(T,0,0)));
end

function forward(loss::Fp50LossHinge,X::Matrix,Y::Vector)
  # @assert size(X,1)==1 "input to fp50loss.forward has to be column matrix with one column"
  posMask = Y.==2;
  negMask = Y.==1;
  mn = mean(X[posMask]);
  O = X[negMask]-mn-1;
  O[O.<0]=0.0;
  return(mean(O));
end

function gradient(loss::Fp50LossHinge,X::Matrix,Y::Vector)
  # @assert size(X,1)==1 "input to fp50loss.gradient has to be column matrix with one column"
  posMask = Y.==2;
  negMask = Y.==1;
  N = sum(negMask);
  P = sum(posMask);
  mn = mean(view(X,:,posMask));
  f=0.0;
  g=zeros(size(X));
  @inbounds for i in find(negMask)
    o=X[i]-mn-1;
    f+=(o>0)?o:0.0;
    g[i]=(o>0)?1.0/N:0;
  end
  f/=N;
  g[posMask] = -mean(view(g,:,negMask))*N/P;
  return (f,g)
end


function backprop!{T}(loss::Fp50LossHinge,X::AbstractArray{T,2},Y::AbstractArray{Int})
  @assert size(X,1)==1 "X has to be column matrix with one column"
  checksize!(X,loss);
  gX=view(loss.gX,:,1:length(Y));
  posMask = Y.==2;
  negMask = Y.==1;
  N = sum(negMask);
  P = sum(posMask);
  mn = mean(view(X,:,posMask));
  f=0.0;
  @inbounds for i in find(negMask)
    o=X[i]-mn-1;
    f+=(o>0)?o:0;
    gX[i]=(o>0)?1.0/N:0;
  end
  f/=N;
  gX[posMask] = -mean(view(gX,:,negMask))*N/P;
  return(f,gX)
end

function checksize!(X::AbstractArray,loss::Fp50LossHinge)
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end