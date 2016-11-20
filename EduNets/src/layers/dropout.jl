export Dropout

type Dropout{T<:AbstractModel}<:AbstractModel
    first::T;
    p::Float64;
    mask::BitVector;
    rng::MersenneTwister;
end

function Dropout(first::AbstractModel,p::Float64;seed::Int=1)
    if (p<=0) || (p>1)
        error("Dropout::Dropout  the probability p hase to be in interval [0,1]")    
    end
    return(Dropout(first,p,BitVector(size(first,1)),MersenneTwister(seed)));
end

@inline function size(model::Dropout)
    return(size(model.first));
end

@inline function size(model::Dropout,k::Int)
    return(size(model.first,k));
end

@inline function update!(model::Dropout,theta::AbstractArray;offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    return offset;
end

@inline function model2vector!(model::Dropout,theta::AbstractArray;offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    return offset;
end

@inline function add!(model::Dropout,theta::AbstractArray;offset::Int=1)
    offset=add!(model.first,theta;offset=offset);
    return offset;
end

@inline function model2vector(model::Dropout)
    return(model2vector(model.first))
end

@inline function forward(model::Dropout,x,v...)
    model.mask=rand(rng,size(x,1)).>model.p;
    x[model.mask,:]=0;
    return(forward(model.first,x,v...));
end

@inline function forward!(model::Dropout,x,v...)
    model.mask=rand(rng,size(x,1)).>model.p;
    x[model.mask,:]=0;
    return(forward!(model.first,x,v...));
end

@inline function backprop!(model::Dropout,x::AbstractArray,go::StridedArray,g::Dropout;update=false)
    gX=backprop!(model.first,x,go,g.first,update=update);
    gX[model.mask,:]=0;
    return(gX)
end

@inline function backprop!(model::Dropout,x::AbstractArray,bags::Array{Array{Int,1},1},go::StridedArray,g::Dropout;update=false)
    gX=backprop!(model.first,x,bags::Array{Array{Int,1},1},go,g.first,update=update);
    gX[model.mask,:]=0;
    return(gX)
end

@inline function gradient!(model::Dropout,x::AbstractArray,go::StridedArray,g::Dropout;update=false)
    gradient!(model.first,x,go,g.first,update=update);
end

@inline function gradient!(model::Dropout,x::AbstractArray,bags::Array{Array{Int,1},1},go::StridedArray,g::Dropout;update=false)
    gradient!(model.first,x,bags::Array{Array{Int,1},1},go,g.first,update=update);
end

@inline function gradient(model::Dropout,x::AbstractArray,go::StridedArray;update=false)
    return(gradient(model.first,x,go));
end


@inline function l1regularize!(model::Dropout,g::Dropout,lambda)
  return(l1regularize!(model.first,g.first,lambda));
end