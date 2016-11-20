import Base.length;
import EduNets.model2vector,EduNets.model2vector!,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!
export model2vector,model2vector!,forward!,forward,gradient!

type AutoEncoderState{T<:AbstractFloat}
    O1::AbstractArray{T,2}
    O2::AbstractArray{T,2}
end

type AutoEncoder{A<:AbstractModel,B<:AbstractModel,T<:AbstractFloat}<:AbstractModel
    first::A;
    second::B;
    state::AutoEncoderState{T}
end

function AutoEncoder(first::AbstractModel,second::AbstractModel;T::DataType=Float64)
    zm=zeros(T,0,0);
    return(AutoEncoder(first,second,AutoEncoderState(zm,zm)));
end

function update!(model::AutoEncoder,theta::AbstractArray;offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
end

function model2vector!(model::AutoEncoder,theta::AbstractArray;offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
end

function add!(model::AutoEncoder,theta::AbstractArray;offset::Int=1)
    offset=add!(model.first,theta;offset=offset);
    offset=add!(model.second,theta;offset=offset);
end

function model2vector(model::AutoEncoder)
    vcat(model2vector(model.first),model2vector(model.second))
end

""" forward(ds::DoubleBagDataset,model::AutoEncoder) """
@inline function forward!(model::AutoEncoder,x::StridedMatrix)
    O1=forward!(model.first,x);
    O2=forward!(model.second,O1);
    return(O2);
end

""" reluMaxReluMaxErrG(ds::DoubleBagDataset,theta::AbstractArray{T,1},k::Tuple{Int,Int,Int})
Returns tuple (f,gW2,gB2,gW1,gB1,gW0,gB0), where f is the error of the network calculated by the hinge loss, and the rest are gradients."""
@inline function gradient!(model::AutoEncoder,x::StridedMatrix,g::AutoEncoder;update::Bool=false)
    #some conversion of parameters and allocation of space for the gradient
    O1=forward!(model.first,x);
    O2=forward!(model.second,O1);

    gO2=(O2-x);
    f=sum(gO2.^2)/size(x,2);
    gO2/=size(x,2);
    #derivative of the output linear unit
    gO1=backprop!(model.second,O1,gO2,g.second;update=update);
    gradient!(model.first,x,gO1,g.first;update=update);
    return(0.5*f)
end