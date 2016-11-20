import Base.length;
using Base.CartesianRange;

export LqPoolingLayer,length,model2vector!,update!,backprop!,gradient!

"""Pooling layer from Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks"""
type LqPoolingLayer{T<:AbstractFloat}<:AbstractLayer
    q::Array{T,1}

    gX::Array{T,2}
    O::Array{T,2}
end

"""function LqPoolingLayer{T}(k::Int;q::T=2.0)
    k --- number of input units (the number of output units is the same)
    q --- starting exponent of the norm pooling (default 2) 
"""
function LqPoolingLayer(k::Int;q::AbstractFloat=2.0,T::DataType=Float64)
    if q<=1
        error("LqPoolingLayer: the q parameter has to be greater than one")
    end
    return(LqPoolingLayer(fill(T(q),k),zeros(T,0,0),zeros(T,0,0)));
end

function length(layer::LqPoolingLayer)
    return(length(layer.q))
end

function model2vector(layer::LqPoolingLayer)
    return(log(exp(layer.q-1)-1));
end

function model2vector!(layer::LqPoolingLayer,theta::AbstractArray;offset::Int=1)
    # Base.LinAlg.copy!(theta,offset,layer.q,1,length(layer.q));
    for i in 1:length(layer.q)
        theta[offset+i-1]=log(exp(layer.q[i]-1)-1)
    end
    return(offset+length(layer));
end

function init!(layer::LqPoolingLayer,X::StridedMatrix)
end

"""
function update!(layer::LqPoolingLayer,theta::AbstractArray;offset::Int=1)
Update of the internal model of q. The update is processed through 1+log(1+exp(q)) 
to ensure that the exponent is greater than 1"""
function update!(layer::LqPoolingLayer,theta::AbstractArray;offset::Int=1)
    for i in 1:length(layer.q)
        layer.q[i]=1+log(1+exp(theta[offset+i-1]));
    end
    # Base.LinAlg.copy!(layer.q,1,theta,offset,length(layer.q))
    return(offset+length(layer));
end


function forward!{T<:AbstractFloat}(layer::LqPoolingLayer{T},X::StridedMatrix{T},bags::Bags)
    if size(layer.O,2)<length(bags)
        layer.O=zeros(T,length(layer.q),length(bags))
    end
    #iterate over bags and biases
    O=view(layer.O,:,1:length(bags))
    fill!(O,0.0);
    for i in 1:length(bags) #iterate over bags
        lb=length(bags[i]);
        for j in bags[i]  #iterate over items (vectors) in bags
            @simd for l in 1:size(X,1)
                @inbounds O[l,i]+=abs(X[l,j])^layer.q[l]
            end
        end
        @simd for l in 1:size(X,1)
            @inbounds O[l,i]=(O[l,i]/lb)^(1.0/layer.q[l])
        end
    end
    return(view(layer.O,:,1:length(bags)));
end

function backprop!{T<:AbstractFloat}(layer::LqPoolingLayer{T},X::StridedMatrix{T},bags::Bags,gO::StridedMatrix{T},g::LqPoolingLayer{T};update=false)
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
    @inbounds for i in eachindex(bags) #iterate over bags
        bag=bags[i]
        lb=length(bag);
        for l in 1:size(X,1)
            if abs(gO[l,i])==0
                continue
            end
            sumq=0.0;
            sumlogx=0.0;
            for j in bag  #iterate over items (vectors) in bags
                if abs(X[l,j])>0
                    xq=abs(X[l,j])^q[l];
                    sumq+=xq
                    sumlogx+=log(abs(X[l,j]))*xq
                    gX[l,j]=q[l]*abs(X[l,j])^(q[l]-1)
                end
            end
            if sumq>0 
                sumq/=lb
                dxsumq=sumq^(1.0/q[l] - 1);
                dxsumq/=q[l];

                sumlogx/=lb;
                g.q[l]+=gO[l,i]*(sumq^(1.0/q[l]))*(-log(sumq)/q[l] + sumlogx/sumq)/q[l]
                for j in bag  
                    gX[l,j]*=sign(X[l,j])*dxsumq*gO[l,i]/lb
                end
            end
        # finish the calculation of derivatives with respect to x
        end
    end

    #the part below is needed to derive the 1+log(1+exp(x)) pre-processing done in the update! and 
    # the processing log(exp(q-1)-1) done ine the model2vector. 
    qq=log(exp(q-1)-1);
    g.q=g.q.*exp(qq)./(1+exp(qq));
    g.q=1+log(1+exp(g.q));

    return(gX);
end

@inline function l1regularize!{T<:AbstractFloat}(model::LqPoolingLayer{T},g::LqPoolingLayer{T},lambda::T)
  return(T(0))
end