import Base.LinAlg.scale!,Base.LinAlg.scale;
export ScalingLayer, scale,scale!;

type ScalingLayer{T}<:AbstractLayer
    mn::Vector{T} #this will be subtracted
    sd::Vector{T} #this will be the divisiaon
end

function ScalingLayer{T}(x::Matrix{T},option="domain";tol=1e-6)
    if option=="domain"
        mn=squeeze(minimum(x,2),2);
        mx=squeeze(maximum(x,2),2);
        sd=mx-mn;
    end
    if option=="variance"
        mn=squeeze(mean(x,2),2);
        sd=squeeze(std(x,2),2);
    end
    
    i=findn(sd.>tol);
    sd[i]=1./sd[i];
    sd[sd.==0]=1;
    return ScalingLayer(mn,sd)
end

function scale!{T}(x::Matrix{T},sc::ScalingLayer{T})
    if size(x,1)!=length(sc.mn)
        error("scale: Scaling parameters and x have different dimensions")
    end
    broadcast!(-,x,x,sc.mn)
    broadcast!(*,x,x,sc.sd)
end

function forward!{T}(sc::ScalingLayer{T},x::StridedMatrix{T})
    if size(x,1)!=length(sc.mn)
        error("scale: Scaling parameters and x have different dimensions")
    end
    broadcast!(-,x,x,sc.mn)
    broadcast!(*,x,x,sc.sd)
    return(x)
end

function forward!(sc::ScalingLayer,ds::AbstractDataset)
    if size(ds.x,1)!=length(sc.mn)
        error("scale: Scaling parameters and ds.x have different dimensions")
    end
    broadcast!(-,ds.x,ds.x,sc.mn)
    broadcast!(*,ds.x,ds.x,sc.sd)
    return(ds)
end

function ScalingLayer(filename::AbstractString;T::DataType=Float64)
    s=readdlm(filename,T);
    return(ScalingLayer{T}(s[:,1],s[:,2]));
end
