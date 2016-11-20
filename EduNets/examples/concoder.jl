import Base.length;
import EduNets.model2vector,EduNets.model2vector!,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!
export model2vector,model2vector!,forward!,forward,gradient!,ConCoder,code!

type ConCoder<:AbstractModel
    first::AbstractModel;
    second::AbstractModel;
end

function update!(model::ConCoder,theta::AbstractArray;offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
end

function model2vector!(model::ConCoder,theta::AbstractArray;offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
end

function add!(model::ConCoder,theta::AbstractArray;offset::Int=1)
    offset=add!(model.first,theta;offset=offset);
    offset=add!(model.second,theta;offset=offset);
end

function model2vector(model::ConCoder)
    vcat(model2vector(model.first),model2vector(model.second))
end

""" forward(ds::DoubleBagDataset,model::ConCoder) """
function forward!(model::ConCoder,x::StridedMatrix)
    O1=forward(model.first,x);
    O2=forward(model.second,O1);
    return(O2);
end

""" forward(ds::DoubleBagDataset,model::ConCoder) """
function forward!(model::ConCoder,x::StridedMatrix)
    O1=forward!(model.first,x);
    O2=forward!(model.second,O1);
    return(O2);
end

""" forward(ds::DoubleBagDataset,model::ConCoder) """
function code!(model::ConCoder,x::StridedMatrix)
    O1=forward!(model.first,x);
    return(O1);
end

""" reluMaxReluMaxErrG(ds::DoubleBagDataset,theta::AbstractArray{T,1},k::Tuple{Int,Int,Int})
Returns tuple (f,gW2,gB2,gW1,gB1,gW0,gB0), where f is the error of the network calculated by the hinge loss, and the rest are gradients."""
function gradient!(model::ConCoder,x::StridedMatrix,g::ConCoder;update::Bool=false)
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