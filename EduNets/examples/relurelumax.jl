import Base.length, Base.size;
import EduNets.model2vector,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!,EduNets.l1regularize!

type ReluReluMaxModel{T<:AbstractFloat}<:AbstractModel
    first::ReluLayer{T};
    second::ReluMaxLayer{T};
    third::LinearLayer{T};
    loss::AbstractLoss;
end

function ReluReluMaxModel(k::Tuple{Int,Int,Int};loss::AbstractLoss=HingeLoss(),T::DataType=Float64)
    return(ReluReluMaxModel(ReluLayer((k[1],k[2]);T=T),ReluMaxLayer((k[2],k[3]);T=T),LinearLayer(k[3];T=T),loss))
end


function size(model::ReluReluMaxModel)
    return(size(model.first,1),size(model.third,2));
end

function size(model::ReluReluMaxModel,i::Int)
    if i==1
        return(size(model.first,1));
    end
    if i==2
        return(size(model.third,2));
    end
    error("ReluReluMaxModel has only dimension 2")
end

function update!{T}(model::ReluReluMaxModel{T},theta::AbstractArray{T};offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
    offset=update!(model.third,theta;offset=offset)
end

function model2vector!{T}(model::ReluReluMaxModel{T},theta::AbstractArray{T};offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
    offset=model2vector!(model.third,theta;offset=offset);
end

function model2vector(model::ReluReluMaxModel)
    vcat(model2vector(model.first),model2vector(model.second),model2vector(model.third))
end

""" loss(ds::SingleBagDataset{T},theta::AbstractArray{T,1},k::Tuple{Int,Int,Int})
Returns error on bags of the network calculated by the hinge loss."""
function loss{T}(model::ReluReluMaxModel{T},ds::SingleBagDataset{T})
    O0=forward(model,ds)
    f=forwas(model.loss,O0,ds.y);
    return(f)
end

""" forward{T}(ds::SingleBagDataset{T},model::ReluReluMaxModel{T}) """
@inline function forward!{T}(model::ReluReluMaxModel{T},ds::SingleBagDataset{T})
    #do the forward propagation
    return(forward!(model,ds.x,ds.bags))
end

""" forward{T}(ds::SingleBagDataset{T},model::ReluReluMaxModel{T}) """
function forward!{T}(model::ReluReluMaxModel{T},x::StridedMatrix,bags::Array{Array{Int,1},1})
    #do the forward propagation
    O1=forward!(model.first,x);
    O2=forward!(model.second,O1,bags);
    O3=forward!(model.third,O2);
    return(O3)
end


""" gradient{T}(ds::SingleBagDataset{T},theta::AbstractArray{T,1},k::Tuple{Int,Int,Int})
Returns tuple (f,gW2,gB2,gW1,gB1,gW0,gB0), where f is the error of the network calculated by the hinge loss, and the rest are gradients."""
function gradient!(model::ReluReluMaxModel,ds::SingleBagDataset,g::ReluReluMaxModel)
    #some conversion of parameters and allocation of space for the gradient
    O1=forward!(model.first,ds.x);
    O2=forward!(model.second,O1,ds.bags);
    O3=forward!(model.third,O2);

    (f,gO3)=gradient!(model.loss,O3,ds.y);
    #derivative of the output linear unit
    gO2=backprop!(model.third,O2,gO3,g.third)
    gO1=backprop!(model.second,O1,gO2,g.second);
    gradient!(model.first,ds.x,gO1,g.first);
    return(f)
end

function l1regularize!(model::ReluReluMaxModel,g::ReluReluMaxModel;lambda=1e-6)
  f=0.0;
  f+=EduNets.l1regularize!(model.first,g.first,lambda);
  f+=EduNets.l1regularize!(model.second,g.second,lambda);
  f+=EduNets.l1regularize!(model.third,g.third,lambda);
  return(f)
end