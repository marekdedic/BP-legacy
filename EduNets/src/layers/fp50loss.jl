
export forward,gradient,Fp50Loss;


type Fp50Loss{T}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function Fp50Loss(;T::DataType=Float64)
  return(Fp50Loss(zeros(T,0,0)));
end

function forward(loss::Fp50Loss,X::Matrix,Y::Vector)
  # @assert size(X,1)==1 "input to fp50loss.forward has to be column matrix with one column"
  posMask = Y.==2;
  negMask = Y.==1;
  mn = mean(X[posMask]);
  O = log(1 + exp(X[negMask]-mn));
  return(mean(O));
end

function gradient(loss::Fp50Loss,X::Matrix,Y::Vector)
  # @assert size(X,1)==1 "input to fp50loss.gradient has to be column matrix with one column"
  posMask = Y.==2;
  negMask = Y.==1;
  N = sum(negMask);
  P = sum(posMask);
  mn = mean(view(X,:,posMask));
  O = exp(view(X,:,negMask)-mn);
  f = mean(log(1 + O));
  g = zeros(length(Y),1);
  g[negMask] = (O./(1+O))/N;
  g[posMask] = -mean(view(g,:,negMask))*N/P;
  return (f,g)
end


function backprop!{T}(loss::Fp50Loss,X::AbstractArray{T,2},Y::AbstractArray{Int})
  @assert size(X,1)==1 "X has to be column matrix with one column"
  checksize!(X,loss);
  gX=view(loss.gX,:,1:length(Y));
  posMask = Y.==2;
  negMask = Y.==1;
  N = sum(negMask);
  P = sum(posMask);
  mn = mean(view(X,:,posMask));
  O = exp(view(X,:,negMask)-mn);
  
  f = mean(log(1 + O));
  gX[negMask] = (O./(1+O))/N;
  gX[posMask] = -mean(view(gX,:,negMask))*N/P;
  return(f,gX)
end

function checksize!(X::AbstractArray,loss::Fp50Loss)
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end