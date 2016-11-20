import Base.length;
import EduNets.model2vector,EduNets.model2vector!,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!,EduNets.l1regularize!

export model2vector,model2vector!,forward!,gradient!,SingleBagModel,l1regularize!

type SingleBagState{T<:AbstractFloat}
    O1::StridedMatrix{T}
    O2::StridedMatrix{T}
    O3::StridedMatrix{T}
end

type SingleBagModel{A<:EduNets.AbstractModel,B<:EduNets.AbstractModel,C<:EduNets.AbstractModel,T<:AbstractFloat,L<:AbstractLoss}<:EduNets.AbstractModel
    first::A;   #this corresponds to the coding layer before the aggregation
    pooling::B; #here, we have the pooling layer
    second::C;  #this is the reconstruction layer that should predict the sample
    loss::L;

    state::SingleBagState{T}
end

function SingleBagState(;T::DataType=Float64)
    zm=zeros(T,0,0)
    return(SingleBagState(zm,zm,zm))
end

function SingleBagModel(A::EduNets.AbstractModel,B::EduNets.AbstractModel,C::EduNets.AbstractModel,loss::EduNets.AbstractLoss;T::DataType=Float64)
    SingleBagModel(A,B,C,loss,SingleBagState(T=T))
end

function init!(model::SingleBagModel,ds::SingleBagDataset)
    init!(model.first,ds.x)
    o=forward!(model.first,ds.x);
    init!(model.pooling,o)
    o=forward!(model.pooling,o,ds.bags)
    init!(model.second,o)
end

function update!{T}(model::SingleBagModel,theta::AbstractArray{T};offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.pooling,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
end

function model2vector!{T}(model::SingleBagModel,theta::AbstractArray{T};offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.pooling,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
end

function model2vector(model::SingleBagModel)
    vcat(model2vector(model.first),model2vector(model.pooling),model2vector(model.second))
end

function forward!(model::SingleBagModel,ds::EduNets.SingleBagDataset)
    return(forward!(model,ds.x,ds.bags));
end

@inline function forward!(model::SingleBagModel,x::StridedMatrix,bags::Array{Array{Int,1},1})
    state=model.state;
    state.O1=forward!(model.first,x);
    state.O2=forward!(model.pooling,state.O1,bags);
    state.O3=forward!(model.second,state.O2);
    return(state.O3);
end

@inline function gradient!(model::SingleBagModel,ds::EduNets.SingleBagDataset,g::SingleBagModel;update::Bool=false)
    state=model.state;
    #some conversion of parameters and allocation of space for the gradient
    forward!(model,ds)
    (f,gO3)=gradient!(model.loss,state.O3,ds.y)
    gradient!(model,ds,gO3,g,update=update)
    return(f)
end

@inline function gradient!(model::SingleBagModel,ds::EduNets.SingleBagDataset,gO::StridedMatrix,g::SingleBagModel;update::Bool=false)
    state=model.state;
    #derivative of the output linear unit
    gO2=backprop!(model.second,state.O2,gO,g.second;update=update);
    gO1=backprop!(model.pooling,state.O1,ds.bags,gO2,g.pooling;update=update);
    gradient!(model.first,ds.x,gO1,g.first;update=update);
end

function l1regularize!(model::SingleBagModel,g::SingleBagModel,lambda::AbstractFloat)
    f=l1regularize!(model.first,g.first,lambda)
    f+=l1regularize!(model.pooling,g.pooling,lambda)
    f+=l1regularize!(model.second,g.second,lambda)
    return(f)
end

@inline function logl1regularize!{A,B,C,L,T}(model::SingleBagModel{A,B,C,L},g::SingleBagModel{A,B,C,L},lambda::T,epsilon::T;dimension=2)
  f=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon,dimension=dimension);
  f+=EduNets.logl1regularize!(model.pooling,g.pooling,lambda,epsilon,dimension=dimension);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon,dimension=dimension);
  return(f)
end