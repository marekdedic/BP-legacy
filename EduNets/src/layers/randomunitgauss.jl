import Base.length;

export RandomUnitGaussLayer,forward,forward!,backprop,backprop!,gradient,gradient!,priorregularization

"""
RandomUnitGaussLayer implements the random Gaussian layer, as has been proposed in
Auto-Encoding Variational Bayes, Diederik P. Kingma and Max Welling, 2014
"""
type RandomUnitGaussLayer{T<:AbstractFloat}<:AbstractMatMulLayer
    O::Matrix{T};   
    epsilon::Matrix{T};   
    gX::Matrix{T};   
end

function RandomUnitGaussLayer(;T::DataType=Float64)
  return(RandomUnitGaussLayer{T}(zeros(T,0,0),zeros(T,0,0),zeros(T,0,0)));
end


function length(model::RandomUnitGaussLayer)
    return(0)
end

function size(model::RandomUnitGaussLayer)
    return((model.k,model.k));
end

function init!(layer::RandomUnitGaussLayer,X::StridedMatrix)
end


@inline function model2vector(model::RandomUnitGaussLayer)
  return(zeros(eltype(model.O),0));
end

@inline function model2vector!(model::RandomUnitGaussLayer,theta::AbstractArray;offset::Int=1)
    return(offset);
end

@inline function add!(model::RandomUnitGaussLayer,theta::AbstractArray;offset::Int=1)
    return(offset);
end

@inline function update!(model::RandomUnitGaussLayer,theta::AbstractArray;offset::Int=1)
    return(offset)
end

@inline function l1regularize!(model::RandomUnitGaussLayer,g::RandomUnitGaussLayer,lambda::AbstractFloat)
  return(zero(eltype(model.O)))
end

""" Regularizes by a log of L1 norm of rows / columns specified by the dimension dimension( 2 --- rows, 1--- columns) """
@inline function logl1regularize!(model::RandomUnitGaussLayer,g::RandomUnitGaussLayer,lambda::AbstractFloat,epsilon::AbstractFloat;dimension=2)
  return(zero(eltype(model.O)))
end

"""
function forward!{T}(model::RandomUnitGaussLayer{T},X::AbstractArray{T})
  let's denote mu=X[1:div(k,2),:] and sigma=X[div(k,2)+1:end,:]
  Then the output of the layer is O=epsilon.*sigma + mu,
  where epsilon  is randomly generated according to N(0,1).
  The epsilon is stored in the neuron, such that derivatives can be correctly calculated
"""
function forward!{T}(model::RandomUnitGaussLayer{T},X::AbstractArray{T})
  #check the size of the cache to store O
  k=size(X,1)
  if size(model.O,1)<k || size(model.O,2)<size(X,2)
    model.O=zeros(T,k,size(X,2));
    model.epsilon=zeros(T,k,size(X,2));
  end

  O=view(model.O,1:k,1:size(X,2))
  epsilon=view(model.epsilon,1:k,1:size(X,2))
  randn!(epsilon)
  for j in 1:size(X,2)
    for i in 1:k
      O[i,j]=epsilon[i,j]+X[i,j]
    end
  end
  
  return(O)
end

"""
function forward!{T}(model::RandomUnitGaussLayer{T},X::AbstractArray{T},nitems::Array{Int,1})
  let's denote mu=X[1:div(k,2),:] and sigma=X[div(k,2)+1:end,:]
  Then the output of the layer is O=epsilon.*sigma + mu,
  where epsilon  is randomly generated according to N(0,1).
  The epsilon is stored in the neuron, such that derivatives can be correctly calculated
"""
function forward!{T}(model::RandomUnitGaussLayer{T},X::AbstractArray{T},nitems::Array{Int,1})
  if length(nitems)!=size(X,2)
    error("the number of vectors in X and the length of the nitems has to be the same")
  end
  #check the size of the cache to store O
  k=size(X,1)
  l=sum(nitems)
  if size(model.O,1)<k || size(model.O,2)<l
    model.O=zeros(T,k,l);
    model.epsilon=zeros(T,k,l);
  end

  O=view(model.O,1:k,1:l)
  epsilon=view(model.epsilon,1:k,1:l)
  randn!(epsilon)
  idx=1;
  bags=Array{UnitRange{Int64},1}(length(nitems))
  for (j,n) in enumerate(nitems)
    lastidx=idx
    for l in 1:n
      for i in 1:k
        O[i,idx]=epsilon[i,idx]+X[i,j]
      end
      idx+=1
    end
    bags[j]=lastidx:idx-1
  end
  return(O,bags)
end


"""
function backprop!{T}(model::RandomUnitGaussLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},g::RandomUnitGaussLayer{T};update::Bool=false)
  let's denote mu=X[1:div(k,2),:] and sigma=X[div(k,2)+1:end,:]
  Then the output of the layer is the gradient with respect to the sigma and mu (gX), when 
  O=epsilon.*sigma + mu. Epsilon  is randomly generated according to N(0,1) during the forward pass and internally stored in the 
  model.
"""
function backprop!{T}(model::RandomUnitGaussLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},g::RandomUnitGaussLayer{T};update::Bool=false)
  #check the size for storing the gradient of gX
  if size(model.gX,1)<size(X,1) || size(model.gX,2)<size(X,2)
    model.gX=zeros(T,size(X));
  end
  k=size(X,1)
  if !update
    fill!(model.gX,0)
  end

  for I in CartesianRange(size(X))
    model.gX[I]+=gO[I]
  end
  gX=view(model.gX,1:size(X,1),1:size(X,2));
  return(gX)
end

"""
function backprop!{T}(model::RandomUnitGaussLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},bags::Array{UnitRange{Int64},1},g::RandomUnitGaussLayer{T};update::Bool=false)
  let's denote mu=X[1:div(k,2),:] and sigma=X[div(k,2)+1:end,:]
  Then the output of the layer is the gradient with respect to the sigma and mu (gX), when 
  O=epsilon.*sigma + mu. Epsilon  is randomly generated according to N(0,1) during the forward pass and internally stored in the 
  model.
"""
function backprop!{T}(model::RandomUnitGaussLayer{T},X::StridedMatrix{T},gO::StridedMatrix{T},bags::Array{UnitRange{Int64},1},g::RandomUnitGaussLayer{T};update::Bool=false)
  #check the size for storing the gradient of gX
  if size(model.gX,1)<size(X,1) || size(model.gX,2)<size(X,2)
    model.gX=zeros(T,size(X));
  end

  k=size(X,1)
  if !update
    fill!(model.gX,0)
  end
  for (bi,bag) in enumerate(bags)
    for j in bag
      for i in 1:k
        model.gX[i,bi]+=gO[i,j]
      end
    end
  end
  gX=view(model.gX,1:size(X,1),1:size(X,2));
  return(gX)
end


"""
  function priorregularization(model::RandomUnitGaussLayer,X::StridedMatrix{T};update::Bool=true)
  This function returns D_{KL}(q(z|x)||p(z)), where p(z)~N(O,I), and q(z|x) is approximately Gaussian, as used above,
  
  The function returns the value of the KL divergence and the gradient.
  If update is true, the regularization is add to the current model.gX
"""
function priorregularization{T}(model::RandomUnitGaussLayer{T},X::StridedMatrix{T};update::Bool=false)
  if !update
    fill!(model.gX,0);
  end

  k=size(X,1)
  l=size(X,2)
  f=zero(T)
  epsilon=T(1e-16)  #a small bound from zero
  for i in 1:size(X,2)
    for j in 1:k
      #increase the value of the optimization function
      f+=X[j,i]^2
      #calculate gradients with respect to mean and standard deviation
      model.gX[j,i]+=2*X[j,i]/l
    end
  end
  f/=l
  return(f,view(model.gX,1:size(X,1),1:size(X,2)))
end