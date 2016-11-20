import Base.length;
using Base.CartesianRange;

export GaussPoolingLayer,length,model2vector!,update!,backprop!,gradient!,init!

"""GaussPoolingLayer implements adaptive pooling function by means of mixture of gaussion. For each dimension,
the outputis calculated as o=\sum_{i=1}^l\sum_{j=1}^k\alpha_jl\exp(-\gamma(x_i-\mu_j)^2 ) 

    Parameters are:
    mu --- centers of gaussian kernels
    gamma --- multipliers of each gaussian center 
    alpha --- multipliers of gaussians
"""
type GaussPoolingLayer{T<:AbstractFloat}<:AbstractLayer
    alpha::Matrix{T} #this is the multipliers on each bin

    mu::Matrix{T} # position of the bin
    gamma::Matrix{T} # width of the bin

    gX::Matrix{T}
    O::Matrix{T}
end

""" 
function GaussPoolingLayer(k::Int;n::Int=10,gamma::AbstractFloat=0.5,T::DataType=Float64)
    k --- input dimension (and output dimension as well)
    n --- number of Gaussians per dimension
    gamma --- width of Gaussian
    T --- DataType
"""
function GaussPoolingLayer(k::Int;n::Int=10,gamma::AbstractFloat=0.5,T::DataType=Float64)
    mu=randn(T,n,k)
    gm=fill(T(gamma),size(mu));
    alpha=fill(one(T)/size(mu,1),size(mu));
    return(GaussPoolingLayer(alpha,mu,gm,zeros(T,0,0),zeros(T,0,0)));
end

""" 
function GaussPoolingLayer(k::Int;n::Int=10,gamma::AbstractFloat=0.5,T::DataType=Float64)
    k --- input dimension (and output dimension as well)
    n --- number of Gaussians per dimension
    gamma --- width of Gaussian
    T --- DataType
"""
function GaussPoolingLayer(k::Int,n::Int;gamma::AbstractFloat=0.5,T::DataType=Float64)
    mu=randn(T,n,k)
    gm=fill(T(gamma),size(mu));
    alpha=fill(one(T)/size(mu,1),size(mu));
    return(GaussPoolingLayer(alpha,mu,gm,zeros(T,0,0),zeros(T,0,0)));
end

"""
function init!(layer::GaussPoolingLayer,X::StridedMatrix)
    Initialize centres of Gaussians to quantiles of the input Matrix X
"""
function init!(layer::GaussPoolingLayer,X::StridedMatrix)
    if size(layer.mu,1)>1
        for k in 1:size(X,1)
            quantile!(view(layer.mu,:,k),X[k,:],linspace(0,1,size(layer.mu,1)))
        end
    else
        layer.mu[:]=median(X,2)
    end
end

"""
function length(layer::GaussPoolingLayer)
    return length of the parameters of the function

"""
function length(layer::GaussPoolingLayer)
    return(length(layer.alpha)+length(layer.mu)+length(layer.gamma))
end


function model2vector{T}(layer::GaussPoolingLayer{T})
  theta=zeros(T,length(layer));
  model2vector!(layer,theta);
  return(theta);
end

function model2vector!(layer::GaussPoolingLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(theta,offset,layer.alpha,1,length(layer.alpha));
    offset+=length(layer.alpha)
    Base.LinAlg.copy!(theta,offset,layer.mu,1,length(layer.mu));
    offset+=length(layer.mu)
    for i in 1:length(layer.gamma)
        theta[offset+i-1]=log(layer.gamma[i];)
    end
    offset+=length(layer.gamma)
    return(offset);
end

function add!(layer::GaussPoolingLayer,theta::AbstractArray;offset::Int=1)
  @simd for i in 1:length(layer.alpha)
    @inbounds theta[offset+i-1]+=layer.alpha[i]
  end
  offset+=length(layer.alpha)
  @simd for i in 1:length(layer.mu)
    @inbounds theta[offset+i-1]+=layer.mu[i]
  end
  offset+=length(layer.mu)
  @simd for i in 1:length(layer.gamma)
    @inbounds theta[offset+i-1]+=log(layer.gamma[i])
  end
  offset+=length(layer.gamma)
  return(offset);
end

function update!(layer::GaussPoolingLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(layer.alpha,1,theta,offset,length(layer.alpha))
    offset+=length(layer.alpha)
    Base.LinAlg.copy!(layer.mu,1,theta,offset,length(layer.mu))
    offset+=length(layer.mu)
    for i in 1:length(layer.gamma)
        layer.gamma[i]=exp(theta[offset+i-1])
    end
    offset+=length(layer.gamma)
    return(offset)
end

function forward!{T<:AbstractFloat}(layer::GaussPoolingLayer{T},X::StridedMatrix{T},bags::Bags)
    if size(layer.O,2)<length(bags)
        layer.O=zeros(T,size(layer.mu,2),length(bags))
    end
    #iterate over bags and biases
    O=view(layer.O,:,1:length(bags))
    fill!(O,zero(T));
    for i in eachindex(bags) #iterate over bags
        for j in bags[i]  #iterate over items (vectors) in bags
            for l in 1:size(X,1)    #iterate over individual dimensions (outputs)
                for a in 1:size(layer.alpha,1) #iterate over the receptive fields
                    O[l,i]+=layer.alpha[a,l]*exp(-layer.gamma[a,l]*(X[l,j]-layer.mu[a,l])^2)
                end
            end
        end
        lb=length(bags[i])
        for l in 1:size(X,1)
            O[l,i]/=lb
        end
    end
    return(O);
end

function backprop!{T<:AbstractFloat}(layer::GaussPoolingLayer{T},X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},g::GaussPoolingLayer{T};update=false)
    if size(layer.gX,2)<size(X,2)
        layer.gX=zeros(T,size(X));
    end
    gX=view(layer.gX,:,1:size(X,2))

    if !update
        fill!(g.mu,0);
        fill!(g.gamma,0);
        fill!(g.alpha,0);
    end
    fill!(gX,0);
    for i in eachindex(bags) #iterate over bags
        for j in bags[i]  #iterate over items (vectors) in bags
            lb=length(bags[i])
            for l in 1:size(X,1)    #iterate over individual dimensions (outputs)
                if gO[l,i]==0
                    continue
                end
                for a in 1:size(layer.alpha,1) #iterate over the receptive fields
                    delta=X[l,j]-layer.mu[a,l];
                    o=exp(-layer.gamma[a,l]*delta^2)
                    g.alpha[a,l]+=o*gO[l,i]/lb
                    g.gamma[a,l]-=delta^2*o*layer.alpha[a,l]*gO[l,i]/lb
                    g.mu[a,l]+=2*delta*layer.alpha[a,l]*o*layer.gamma[a,l]*gO[l,i]/lb
                    layer.gX[l,j]-=2*delta*layer.alpha[a,l]*o*layer.gamma[a,l]*gO[l,i]/lb
                end
            end
        end
    end

    #correction for using exponential in the update!
    g.gamma.*=layer.gamma
    g.gamma=exp(g.gamma)

    return(gX);
end

@inline function l1regularize!{T<:AbstractFloat}(layer::GaussPoolingLayer{T},g::GaussPoolingLayer{T},lambda::T)
    f=lambda*sumabs(layer.alpha)
    g.alpha+=lambda*sign(layer.alpha)
    f+=lambda*sumabs(layer.gamma)
    g.gamma+=lambda*sign(layer.gamma)
    return(f)    
end