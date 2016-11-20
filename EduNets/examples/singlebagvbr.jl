using EduNets
import Base.length;
import EduNets.model2vector,EduNets.model2vector!,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!,EduNets.l1regularize!
# export model2vector,model2vector!,forward!,gradient!,SingleBagVBRModel,l1regularize!

type SingleBagVBRState{T<:AbstractFloat}
    O1::StridedMatrix{T}
    O2::StridedMatrix{T}
    O3::StridedMatrix{T}
    O4::StridedMatrix{T}
    bags::Array{UnitRange{Int64},1}
end

type SingleBagVBRModel{A<:EduNets.AbstractModel,B<:EduNets.AbstractModel,C<:EduNets.AbstractModel,D<:EduNets.AbstractModel,T<:AbstractFloat,L<:AbstractLoss}<:EduNets.AbstractModel
    first::A;   #this corresponds to the coding layer before the aggregation
    pooling::B; #here, we have the pooling layer
    generating::C; #here, there will be the random layer that will sample bags
    second::D;  #this is the reconstruction layer that should predict the sample
    loss::L;

    state::SingleBagVBRState{T}
end

function SingleBagVBRState(;T::DataType=Float64)
    zm=zeros(T,0,0)
    return(SingleBagVBRState(zm,zm,zm,zm,Array{UnitRange{Int64},1}(0)))
end

function SingleBagVBRModel(A::EduNets.AbstractModel,B::EduNets.AbstractModel,C::EduNets.AbstractModel,D::EduNets.AbstractModel,loss::EduNets.AbstractLoss;T::DataType=Float64)
    SingleBagVBRModel(A,B,C,D,loss,SingleBagVBRState(T=T))
end

function update!{T}(model::SingleBagVBRModel,theta::AbstractArray{T};offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.pooling,theta;offset=offset)
    offset=update!(model.generating,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
end

function model2vector!{T}(model::SingleBagVBRModel,theta::AbstractArray{T};offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.pooling,theta;offset=offset);
    offset=model2vector!(model.generating,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
end

function model2vector(model::SingleBagVBRModel)
    vcat(model2vector(model.first),model2vector(model.pooling),model2vector(model.generating),model2vector(model.second))
end

function forward!(model::SingleBagVBRModel,ds::EduNets.SingleBagDataset)
    return(forward!(model,ds.x,ds.bags));
end

"""
    recognize!(model::SingleBagVBRModel,ds::SingleBagDataset)

    project the dataset ds into representation in hidden variables

"""
function recognize!(model::SingleBagVBRModel,ds::SingleBagDataset)
  xx=forward!(model.first,ds.x)
  xx=forward!(model.pooling,xx,ds.bags)
  return(Dataset(xx,ds.y))
end

@inline function forward!(model::SingleBagVBRModel,x::StridedMatrix,bags::Bags)
    state=model.state;
    state.O1=forward!(model.first,x);
    state.O2=forward!(model.pooling,state.O1,bags);
    (state.O3,state.bags)=forward!(model.generating,state.O2,map(length,bags));
    state.O4=forward!(model.second,state.O3);
    return(state.O4,state.bags);
end

@inline function gradient!(model::SingleBagVBRModel,ds::EduNets.SingleBagDataset,g::SingleBagVBRModel;update::Bool=false)
    state=model.state;
    #some conversion of parameters and allocation of space for the gradient
    forward!(model,ds)
    (fl,gO4)=gradient!(model.loss,state.O4,state.bags,ds.x,ds.bags)
    gO3=backprop!(model.second,state.O3,gO4,g.second;update=update);
    gO2=backprop!(model.generating,state.O2,gO3,state.bags,g.generating;update=update);
    (fr,gO2)=priorregularization(model.generating,state.O2;update=true);
    gO1=backprop!(model.pooling,state.O1,ds.bags,gO2,g.pooling;update=update);
    gradient!(model.first,ds.x,gO1,g.first;update=update);
    return(fl+fr)
end

function l1regularize!(model::SingleBagVBRModel,g::SingleBagVBRModel,lambda::AbstractFloat)
    f=l1regularize!(model.first,g.first,lambda)
    f+=l1regularize!(model.pooling,g.pooling,lambda)
    f+=l1regularize!(model.second,g.second,lambda)
    return(f)
end

function testSingleBagVBRModel()
  x=randn(10,100);
  y=rand(1:2,100);
  ds=SingleBagDataset(x,y,rand(1:10,100))

  T=Float64
  model=SingleBagVBRModel(
    ReluLayer((10,20);T=T),
    MeanPoolingLayer(20;T=T),
    RandomGaussLayer(;T=T),
    StackedBlocks(
      ReluLayer((10,10);T=T),
      LinearLayer((10,10);T=T),T=T),
    MeanL2BagLoss(T=T),T=T);
  gmodel=deepcopy(model);
  gvec=model2vector(gmodel);
  function fOpt(x)
    srand(1)
    update!(model,x)
    f=gradient!(model,ds,gmodel);
    model2vector!(gmodel,gvec)
    return(f,gvec)
  end
  EduNets.testgradient(fOpt,model2vector(model);verbose=1,h=1e-6);
end

# testSingleBagVBRModel()