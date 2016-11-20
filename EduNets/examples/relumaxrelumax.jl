import Base.length;
import EduNets.model2vector,EduNets.forward,EduNets.forward!,EduNets.forward!,EduNets.update!,EduNets.gradient!
type ReluMaxReluMaxModel{T<:AbstractFloat}<:AbstractModel
    first::ReluMaxLayer{T};
    second::ReluMaxLayer{T};
    third::LinearLayer{T};
    loss::AbstractLoss;
end

function ReluMaxReluMaxModel(k::Tuple{Int,Int,Int};T::DataType=Float64)
    return(ReluMaxReluMaxModel(ReluMaxLayer((k[1],k[2]);T=T),ReluMaxLayer((k[2],k[3]);T=T),LinearLayer(k[3];T=T),HingeLoss(;T=T)))
end

function ReluMaxReluMaxModel(k::Tuple{Int,Int,Int,Int},loss::AbstractLoss;T::DataType=Float64)
    return(ReluMaxReluMaxModel(ReluMaxLayer((k[1],k[2]);T=T),ReluMaxLayer((k[2],k[3]);T=T),LinearLayer((k[3],k[4]);T=T),loss))
end

function update!{T}(model::ReluMaxReluMaxModel{T},theta::AbstractArray{T};offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
    offset=update!(model.third,theta;offset=offset)
end

function model2vector!{T}(model::ReluMaxReluMaxModel{T},theta::AbstractArray{T};offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
    offset=model2vector!(model.third,theta;offset=offset);
end

function model2vector(model::ReluMaxReluMaxModel)
    vcat(model2vector(model.first),model2vector(model.second),model2vector(model.third))
end

""" forward{T}(ds::DoubleBagDataset{T},model::ReluMaxReluMaxModel{T}) """
function forward{T}(model::ReluMaxReluMaxModel{T},ds::DoubleBagDataset{T})
    #do the forward propagation
    (O2,_)=forward(model.first,ds.x,ds.subbags);
    (O1,_)=forward(model.second,O2,ds.bags);
    O0=forward(model.third,O1);
    return(O0)
end

""" forward{T}(ds::DoubleBagDataset{T},model::ReluMaxReluMaxModel{T}) """
function forward!{T}(model::ReluMaxReluMaxModel{T},ds::DoubleBagDataset{T})
    #do the forward propagation
    O2=forward!(model.first,ds.x,ds.subbags);
    O1=forward!(model.second,O2,ds.bags);
    O0=forward!(model.third,O1);
    return(O0)
end

function size(model::ReluMaxReluMaxModel)
    return(size(model.first,1),size(model.third,2));
end

function size(model::ReluMaxReluMaxModel,i::Int)
    if i==1
        return(size(model.first,1));
    end
    if i==2
        return(size(model.third,2));
    end
    error("ReluMaxReluMaxModel has only dimension 2")
end

function loss{T}(model::ReluMaxReluMaxModel{T},ds::DoubleBagDataset{T})
    O0=forward(ds,model)
    f=forward(model.loss,O0,ds.y);
    return(f)
end

function gradient!{T}(model::ReluMaxReluMaxModel{T},ds::DoubleBagDataset{T},g::ReluMaxReluMaxModel{T};update::Bool=false)
    #some conversion of parameters and allocation of space for the gradient
    O1=forward!(model.first,ds.x,ds.subbags);
    O2=forward!(model.second,O1,ds.bags);
    O3=forward!(model.third,O2);

    (f,gO3)=gradient!(model.loss,O3,ds.y);
    #derivative of the output linear unit
    gO2=backprop!(model.third,O2,gO3,g.third;update=update)
    gO1=backprop!(model.second,O1,gO2,g.second;update=update);
    gradient!(model.first,ds.x,gO1,g.first;update=update);
    return(f)
end


"""This is a version for importance sampling. Ensure that w is properly scaled, it should sum up to one, and it should be non zero"""
function gradient!{T}(model::ReluMaxReluMaxModel{T},ds::DoubleBagDataset{T},g::ReluMaxReluMaxModel{T},w::AbstractArray{T,1};update::Bool=false)
    #some conversion of parameters and allocation of space for the gradient
    O1=forward!(model.first,ds.x,ds.subbags);
    O2=forward!(model.second,O1,ds.bags);
    O3=forward!(model.third,O2);

    (f,gO3)=gradient!(model.loss,O3,ds.y,w);
    #derivative of the output linear unit
    gO2=backprop!(model.third,O2,gO3,g.third;update=update)
    gO1=backprop!(model.second,O1,gO2,g.second;update=update);
    gradient!(model.first,ds.x,gO1,g.first;update=update);
    return(f)
end

function explainlayer{T}(maxI::AbstractArray,layer::ReluMaxLayer{T},outw::AbstractArray{T,1})
    #iterate over outputs of each bag
    I=setdiff(unique(maxI),0);
    W=zeros(T,size(layer.W,1),length(I));
    B=zeros(T,length(I));
    idxmap=Dict(zip(I,1:length(I)))
    for ii in CartesianRange(size(maxI))
        s=maxI[ii] #index of the contributing instance
        if s>0
            s=idxmap[s];    #get instance index in W and B
            k=ii[1] #index of the neuron
            W[:,s]+=outw[k]*layer.W[:,k]
            B[s]+=outw[k]*layer.B[k]
        end
    end
    return(W,B,idxmap);
end

function l1regularize!{T}(model::ReluMaxReluMaxModel{T},g::ReluMaxReluMaxModel{T},lambda::T)
  f=eltype(model.first.W)(0.0);
  f+=EduNets.l1regularize!(model.first,g.first,lambda);
  f+=EduNets.l1regularize!(model.second,g.second,lambda);
  f+=EduNets.l1regularize!(model.third,g.third,lambda);
  return(f)
end

function logl1regularize!{T}(model::ReluMaxReluMaxModel{T},g::ReluMaxReluMaxModel{T},lambda::T,epsilon::T)
  f=eltype(model.first.W)(0.0)
  f+=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon;dimension=2);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon;dimension=1);
  return(f)
end


function crosslogl1regularize!{T}(model::ReluMaxReluMaxModel{T},g::ReluMaxReluMaxModel{T},lambda::T,epsilon::T)
  f=eltype(model.first.W)(0.0)
  f+=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon;dimension=1);
  f+=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon;dimension=2);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon;dimension=1);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon;dimension=2);
  return(f)
end



""" function explain(ds::DoubleBagDataset,model::ReluMaxReluMaxModel;onlypositive=false) 
    returns tuple (O,bagidx,subbagidxs,instancecontributions),
    where bagidxs are indexes of bags (samples) sorted in the order from the most positive to the most negative
    subbagidxs are indexes of subbags which as contributed to the detection,
    and instance contributions contains contribution of individual instances 

    idxs contains indexes of bags (user:5min window) and indexes are sorted according to the output of the network.
    ii contains indexes of sub-bags
    xi contains indexes of instances 
    contrib is the contribution of the given instance
    returned 4-tuple (idxs,ii,xi,contrib)
"""
function explain(ds::DoubleBagDataset,model::ReluMaxReluMaxModel;onlypositive::Float64=-Inf,onlynegative=false,nreturned=0)
    #do the forward part of the network
    (O1,maxI1)=forward(model.first,ds.x,ds.subbags);
    (O2,maxI2)=forward(model.second,O1,ds.bags); 
    O3=squeeze(forward(model.third,O2),1);

    idxs=sortperm(O3,rev=true);
    idxs=idxs[O3[idxs].>onlypositive];

    if onlynegative
        idxs=sortperm(O3);
        idxs=idxs[O3[idxs].<0];
    end

    if nreturned>0
        idxs=idxs[1:min(nreturned,length(idxs))];
    end

    explanations=[];
    for idx in idxs
        (W2,B2,idxmap2)=explainlayer(maxI2[:,idx],model.second,squeeze(model.third.W,2));
        OO=zeros(0,4);
        for ii in keys(idxmap2)
            (W1,B1,idxmap1)=explainlayer(maxI1[:,ii],model.first,W2[:,idxmap2[ii]]);
            for xi in keys(idxmap1)
                contrib=dot(ds.x[:,xi],W1[:,idxmap1[xi]])+B1[idxmap1[xi]]+B2[idxmap2[ii]];
    #             @printf("%d %d %d: %f\n",idx,ii,xi,contrib)
                OO=vcat(OO,transpose([idx,ii,xi,contrib]));
            end
        end
        I=sortperm(OO[:,4],rev=true);
        OO=OO[I,:];
        mask=OO[:,4].>0;
        OO=OO[mask,:]
        push!(explanations,OO)
    end
    return(explanations,O3)
end