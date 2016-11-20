import Base.length;
using Base.CartesianRange;

export WedgePoolingLayer,length,model2vector!,update!,backprop!,gradient!

"""Pooling layer from Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks"""
type WedgePoolingLayer{T<:AbstractFloat}<:AbstractLayer
    alpha::Matrix{T} #this is the multipliers on each bin

    left::T # left bound of the interval
    step::T # width of one triangle
    kmax::Int

    gX::Matrix{T}
    O::Matrix{T}
end

function increasebin!{T}(layer::WedgePoolingLayer{T},x::T) 
    
end

"""function WedgePoolingLayer{T}(k::Int;q::T=2.0)
    k --- number of input units (the number of output units is the same)
    q --- starting exponent of the norm pooling (default 2) 
"""
function WedgePoolingLayer(k::Int;left=0.0,right=5,step=0.5,T::DataType=Float64)
    kmax=(right-left)/step
    return(WedgePoolingLayer(fill(one(T),k,kmax),T(left),T(step),kmax,zeros(T,0,0),zeros(T,0,0)));
end

function length(layer::WedgePoolingLayer)
    return(length(layer.alpha))
end

function model2vector(model::WedgePoolingLayer)
  theta=zeros(eltype(model.alpha),length(model.alpha));
  theta[1:length(model.alpha)]=model.alpha;
  return(theta);
end

function model2vector!(model::WedgePoolingLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(theta,offset,model.alpha,1,length(model.alpha));
    return(offset+length(model));
end

function add!(model::WedgePoolingLayer,theta::AbstractArray;offset::Int=1)
  @simd for i in 1:length(model.alpha)
    @inbounds theta[offset+i-1]+=model.alpha[i]
  end
  offset+=length(model.alpha)
  return(offset);
end

function update!(model::AbstractMatMulLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(model.alpha,1,theta,offset,length(model.alpha))
    return(offset+length(model))
end

function forward!{T<:AbstractFloat}(layer::WedgePoolingLayer{T},X::StridedMatrix{T},bags::Bags)
    if size(layer.O,2)<length(bags)
        layer.O=zeros(T,length(layer.q),length(bags))
    end
    #iterate over bags and biases
    O=view(layer.O,:,1:length(bags))
    fill!(O,zero(T));
    for i in 1:length(bags) #iterate over bags
        for j in bags[i]  #iterate over items (vectors) in bags
            for l in 1:size(X,1)
                x=(X[l,j]-layer.left)/layer.step

                xf=floor(x);
                k=(Int)xf+1;
                r=x-xf;

                # do the clipping into the area of allowed numbers
                r=(k<1)?1:r;
                k=(k<1)?0:k;
                r=(k>layer.kmax)?0:r;
                k=(k>layer.kmax)?layer.kmax-1:k;

                # increase the values
                O[l,i]=O[l,i]+alpha[l,k]*r;
                O[l+1,i]=O[l+1,i]+alpha[l,k]*(1-r);
            end
        end
        lb=length(bags[i])
        for l in 1:size(X,1)
            O[l,i]/=lb
        end
    end
    return(O);
end

function backprop!{T<:AbstractFloat}(layer::WedgePoolingLayer{T},X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},g::WedgePoolingLayer{T};update=false)
    if size(layer.gX,2)<size(X,2)
        layer.gX=zeros(T,size(X));
    end

    if !update
        fill!(g.q,0);
    end
    q=layer.q;
    gX=view(layer.gX,:,1:size(X,2));
    fill!(gX,0.0);
    sumq=0.0;
    sumlogx=0.0;
    for i in eachindex(bags) #iterate over bags
        bag=bags[i]
        lb=length(bag);
        for l in 1:size(X,1)
            if abs(gO[l,i])==0
                continue
            end
            x=(X[l,j]-layer.left)/layer.step

            xf=floor(x);
            k=(Int)xf+1;
            r=x-xf;

            # do the clipping into the area of allowed numbers
            r=(k<1)?1:r;
            k=(k<1)?0:k;
            r=(k>layer.kmax)?0:r;
            k=(k>layer.kmax)?layer.kmax-1:k;

            # increase the values
            O[l,i]=O[l,i]+alpha[l,k]*r;
            O[l+1,i]=O[l+1,i]+alpha[l,k]*(1-r);
        end
    end
    return(gX);
end

@inline function l1regularize!{T<:AbstractFloat}(model::WedgePoolingLayer{T},g::WedgePoolingLayer{T},lambda::T)
  return(T(0))
end