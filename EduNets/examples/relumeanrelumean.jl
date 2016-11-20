import Base.length;
import EduNets.model2vector,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!
type ReluMeanReluMeanModel{T<:AbstractFloat}<:AbstractModel
    first::ReluMaxLayer{T};
    second::ReluMaxLayer{T};
    third::LinearLayer{T};
    loss::AbstractLoss;
end

function ReluMeanReluMeanModel(k::Tuple{Int,Int,Int};T::DataType=Float64)
    return(ReluMeanReluMeanModel(ReluMaxLayer(k[1],k[2];T=T),ReluMaxLayer(k[2],k[3];T=T),LinearLayer(k[3];T=T),HingeLoss()))
end

function ReluMeanReluMeanModel(k::Tuple{Int,Int,Int,Int};T::DataType=Float64)
    return(ReluMeanReluMeanModel(ReluMaxLayer(k[1],k[2];T=T),ReluMaxLayer(k[2],k[3];T=T),LinearLayer(k[3],k[4];T=T),MultiHingeLoss(k[4])))
end

function ReluMeanReluMeanModel(k::Tuple{Int,Int,Int},loss::AbstractLoss;T::DataType=Float64)
    return(ReluMeanReluMeanModel(ReluMaxLayer(k[1],k[2];T=T),ReluMaxLayer(k[2],k[3];T=T),LinearLayer(k[3],length(loss);T=T),loss))
end

function update!{T}(model::ReluMeanReluMeanModel{T},theta::AbstractArray{T};offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
    offset=update!(model.third,theta;offset=offset)
end

function model2vector!{T}(model::ReluMeanReluMeanModel{T},theta::AbstractArray{T};offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
    offset=model2vector!(model.third,theta;offset=offset);
end

function model2vector(model::ReluMeanReluMeanModel)
    vcat(model2vector(model.first),model2vector(model.second),model2vector(model.third))
end

""" forward{T}(ds::DoubleBagDataset{T},model::ReluMeanReluMeanModel{T}) """
function forward{T}(model::ReluMeanReluMeanModel{T},ds::DoubleBagDataset{T})
    #do the forward propagation
    (O2,_)=forward(model.first,ds.x,ds.subbags);
    (O1,_)=forward(model.second,O2,ds.bags);
    O0=forward(model.third,O1);
    return(O0)
end

""" reluMaxReluMaxErr(ds::DoubleBagDataset{T},theta::AbstractArray{T,1},k::Tuple{Int,Int,Int})
Returns error on bags of the network calculated by the hinge loss."""
function loss{T}(model::ReluMeanReluMeanModel{T},ds::DoubleBagDataset{T})
    O0=forward(ds,model)
    f=forward(model.loss,O0,ds.y);
    return(f)
end

""" reluMaxReluMaxErrG{T}(ds::DoubleBagDataset{T},theta::AbstractArray{T,1},k::Tuple{Int,Int,Int})
Returns tuple (f,gW2,gB2,gW1,gB1,gW0,gB0), where f is the error of the network calculated by the hinge loss, and the rest are gradients."""
function gradient!{T}(model::ReluMeanReluMeanModel{T},ds::DoubleBagDataset{T},g::ReluMeanReluMeanModel{T};update::Bool=false)
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

function l1regularize!(model::ReluMeanReluMeanModel,g::ReluMeanReluMeanModel;lambda=1e-6)
  f=0.0;
  f+=EduNets.l1regularize!(model.first,g.first,lambda);
  f+=EduNets.l1regularize!(model.second,g.second,lambda);
  f+=EduNets.l1regularize!(model.third,g.third,lambda);
  return(f)
end

function logl1regularize!(model::ReluMeanReluMeanModel,g::ReluMeanReluMeanModel;lambda=1e-6,epsilon=1e-2)
  f=0.0
  f+=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon;dimension=2);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon;dimension=1);
  return(f)
end


function crosslogl1regularize!(model::ReluMeanReluMeanModel,g::ReluMeanReluMeanModel;lambda=1e-6,epsilon=1e-2)
  f=0.0
  f+=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon;dimension=1);
  f+=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon;dimension=2);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon;dimension=1);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon;dimension=2);
  return(f)
end



""" function explain(ds::DoubleBagDataset,model::ReluMeanReluMeanModel;onlypositive=false) 
    returns tuple (O,bagidx,subbagidxs,instancecontributions),
    where bagidxs are indexes of bags (samples) sorted in the order from the most positive to the most negative
    subbagidxs are indexes of subbags which as contributed to the detection,
    and instancecontributions contains contribution of individual instances """
function explain(ds::DoubleBagDataset,model::ReluMeanReluMeanModel;onlypositive=false,onlynegative=false,nreturned=0)
    #do the forward part of the network
    (O1,maxI1)=forward(model.first,ds.x,ds.subbags);
    (O2,maxI2)=forward(model.second,O1,ds.bags); 
    O3=squeeze(forward(model.third,O2),1);

    if onlypositive && onlynegative
        error("ReluMeanReluMeanModel::explain onlypositive and onlynegative does not make sense");
    end

    idxs=sortperm(O3,rev=true);
    if onlypositive
        idxs=idxs[O3[idxs].>0];
    end

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
    return(explanations)
end