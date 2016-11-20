import Base.length;
import EduNets.model2vector,EduNets.model2vector!,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!
export model2vector,model2vector!,forward!,gradient!,MilCoder,code!
type MilCoder{A<:EduNets.AbstractModel,B<:EduNets.AbstractPooling,C<:EduNets.AbstractModel,D<:EduNets.AbstractLoss}<:EduNets.AbstractModel
    first::A;   #this corresponds to the coding layer before the aggregation
    pooling::B; #here, we have the pooling layer
    second::C;  #this is the reconstruction layer that should predict the sample

    loss::D
end

function update!(model::MilCoder,theta::AbstractArray;offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.pooling,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
end

function model2vector!(model::MilCoder,theta::AbstractArray;offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.pooling,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
end

function add!(model::MilCoder,theta::AbstractArray;offset::Int=1)
    offset=add!(model.first,theta;offset=offset);
    offset=add!(model.pooling,theta;offset=offset);
    offset=add!(model.second,theta;offset=offset);
end

function model2vector(model::MilCoder)
    vcat(model2vector(model.first),model2vector(model.pooling),model2vector(model.second))
end

function forward!(model::MilCoder,ds::EduNets.SingleBagDataset)
    return(forward!(model,ds.x,ds.bags));
end

function forward!(model::MilCoder,x::StridedMatrix,bags::AbstractArray{AbstractArray{Int,1},1})
    O1=forward!(model.first,x);
    O2=forward!(model.pooling,O1,bags);
    O3=forward!(model.second,O1);
    return(O3);
end

function code!(model::MilCoder,ds::EduNets.SingleBagDataset)
    return(forward!(model,ds.x,ds.bags));
end

""" forward(ds::DoubleBagDataset,model::MilCoder) """
function code!(model::MilCoder,x::StridedMatrix,bags::Array{Array{Int,1},1})
    O1=forward!(model.first,x);
    O2=forward!(model.pooling,O1,bags);
    return(O2);
end

""" reluMaxReluMaxErrG(ds::DoubleBagDataset,theta::AbstractArray{T,1},k::Tuple{Int,Int,Int})
Returns tuple (f,gW2,gB2,gW1,gB1,gW0,gB0), where f is the error of the network calculated by the hinge loss, and the rest are gradients."""
function gradient!(model::MilCoder,x::StridedMatrix,bags::Array{Array{Int,1},1},y::StridedMatrix,g::MilCoder;update::Bool=false)
    #some conversion of parameters and allocation of space for the gradient
    O1=forward!(model.first,x);
    O2=forward!(model.pooling,O1,bags);
    O3=forward!(model.second,O2);

    # (gO3=(O3-y);
    # f=sum(gO3.^2)/size(y,2);
    # gO3/=size(y,2);)
    (f,gO3)=gradient!(model.loss,O3,y);
    #derivative of the output linear unit
    gO2=backprop!(model.second,O2,gO3,g.second;update=update);
    gO1=backprop!(model.pooling,O1,bags,gO2,g.pooling);
    gradient!(model.first,x,gO1,g.first;update=update);
    return(f)
end