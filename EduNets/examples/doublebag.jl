import Base.length,Base.size;
import EduNets.model2vector,EduNets.model2vector!,EduNets.forward,EduNets.forward!,EduNets.update!,EduNets.gradient!,EduNets.l1regularize!,EduNets.init!

export model2vector,model2vector!,forward!,gradient!,DoubleBagModel,l1regularize!,logl1regularize!,explain,init!

type DoubleBagState{T<:AbstractFloat}
  O1::StridedMatrix{T}
  O2::StridedMatrix{T}
  O3::StridedMatrix{T}
  O4::StridedMatrix{T}
  O5::StridedMatrix{T}
end

function DoubleBagState(T::DataType=Float64)
  zs=zeros(T,0,0);
  DoubleBagState(zs,zs,zs,zs,zs);
end

type DoubleBagModel{A<:EduNets.AbstractModel,B<:EduNets.AbstractModel,C<:EduNets.AbstractModel,D<:EduNets.AbstractModel,E<:EduNets.AbstractModel,T<:AbstractFloat,L<:EduNets.AbstractLoss}<:EduNets.AbstractModel
    first::A;   
    pooling1::B;
    second::C;  
    pooling2::D;
    third::E;  
    loss::L;
    
    state::DoubleBagState{T};
end

function DoubleBagModel(A::EduNets.AbstractModel,B::EduNets.AbstractModel,C::EduNets.AbstractModel,D::EduNets.AbstractModel,E::EduNets.AbstractModel,loss::EduNets.AbstractLoss;T::DataType=Float64)
  DoubleBagModel(A,B,C,D,E,loss,DoubleBagState(T))
end

function init!(model::DoubleBagModel,ds::DoubleBagDataset)
    init!(model.first,ds.x)
    o=forward!(model.first,ds.x);
    init!(model.pooling1,o)
    o=forward!(model.pooling1,o,ds.subbags)
    init!(model.second,o)
    o=forward!(model.second,o)
    init!(model.pooling2,o)
    o=forward!(model.pooling2,o,ds.bags)
    init!(model.third,o)
end

@inline function update!{A,B,C,D,E}(model::DoubleBagModel{A,B,C,D,E},theta::AbstractArray;offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.pooling1,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
    offset=update!(model.pooling2,theta;offset=offset)
    offset=update!(model.third,theta;offset=offset)
    return(offset)
end

@inline function add!{A,B,C,D,E}(model::DoubleBagModel{A,B,C,D,E},theta::AbstractArray;offset::Int=1)
    offset=add!(model.first,theta;offset=offset)
    offset=add!(model.pooling1,theta;offset=offset)
    offset=add!(model.second,theta;offset=offset)
    offset=add!(model.pooling2,theta;offset=offset)
    offset=add!(model.third,theta;offset=offset)
end

@inline function model2vector!{A,B,C,D,E}(model::DoubleBagModel{A,B,C,D,E},theta::AbstractArray;offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.pooling1,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
    offset=model2vector!(model.pooling2,theta;offset=offset);
    offset=model2vector!(model.third,theta;offset=offset);
    return(offset)
end

@inline function model2vector(model::DoubleBagModel)
    vcat(model2vector(model.first),model2vector(model.pooling1),model2vector(model.second),model2vector(model.pooling2),model2vector(model.third))
end

@inline function forward!(model::DoubleBagModel,ds::EduNets.DoubleBagDataset)
    return(forward!(model,ds.x,ds.bags,ds.subbags));
end

@inline function forward!(model::DoubleBagModel,x::StridedMatrix,bags::Array{Array{Int,1},1},subbags::Array{Array{Int,1},1})
    model.state.O1=forward!(model.first,x);
    model.state.O2=forward!(model.pooling1,model.state.O1,subbags);
    model.state.O3=forward!(model.second,model.state.O2);
    model.state.O4=forward!(model.pooling2,model.state.O3,bags);
    model.state.O5=forward!(model.third,model.state.O4);
    return(model.state.O5);
end

@inline function gradient!{A,B,C,D,E,T}(model::DoubleBagModel{A,B,C,D,E,T},ds::EduNets.DoubleBagDataset,g::DoubleBagModel{A,B,C,D,E,T};update::Bool=false)
  #some conversion of parameters and allocation of space for the gradient
  forward!(model,ds)
  (f,gO5)=gradient!(model.loss,model.state.O5,ds.y)
  gradient!(model,ds,gO5,g,update=update)
  return(f)
end

@inline function gradient!{A,B,C,D,E,T}(model::DoubleBagModel{A,B,C,D,E,T},ds::EduNets.DoubleBagDataset,gO5::StridedMatrix{T},g::DoubleBagModel{A,B,C,D,E,T};update::Bool=false)
    #derivative of the output linear unit
    gO4=backprop!(model.third,model.state.O4,gO5,g.third;update=update);
    gO3=backprop!(model.pooling2,model.state.O3,ds.bags,gO4,g.pooling2;update=update);
    gO2=backprop!(model.second,model.state.O2,gO3,g.second;update=update);
    gO1=backprop!(model.pooling1,model.state.O1,ds.subbags,gO2,g.pooling1;update=update);
    gradient!(model.first,ds.x,gO1,g.first;update=update);
end


@inline function l1regularize!{A,B,C,D,E,T}(model::DoubleBagModel{A,B,C,D,E,T},g::DoubleBagModel{A,B,C,D,E,T},lambda::T)
  f=EduNets.l1regularize!(model.first,g.first,lambda);
  f+=EduNets.l1regularize!(model.pooling1,g.pooling1,lambda);
  f+=EduNets.l1regularize!(model.second,g.second,lambda);
  f+=EduNets.l1regularize!(model.pooling2,g.pooling2,lambda);
  f+=EduNets.l1regularize!(model.third,g.third,lambda);
  return(f)
end

@inline function logl1regularize!{A,B,C,D,E,T}(model::DoubleBagModel{A,B,C,D,E,T},g::DoubleBagModel{A,B,C,D,E,T},lambda::T,epsilon::T;dimension=2)
  f=EduNets.logl1regularize!(model.first,g.first,lambda,epsilon,dimension=dimension);
  f+=EduNets.logl1regularize!(model.pooling1,g.pooling1,lambda,epsilon,dimension=dimension);
  f+=EduNets.logl1regularize!(model.second,g.second,lambda,epsilon,dimension=dimension);
  f+=EduNets.logl1regularize!(model.pooling2,g.pooling2,lambda,epsilon,dimension=dimension);
  f+=EduNets.logl1regularize!(model.third,g.third,lambda,epsilon,dimension=dimension);
  return(f)
end

""" 
    function explainlayer{T}(maxI::AbstractArray,layer::ReluMaxLayer{T},outw::AbstractArray{T,1})

"""


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

# function explain(ds::DoubleBagDataset,model::ReluMaxReluMaxModel;onlypositive::Float64=-Inf,onlynegative=false,nreturned=0)
#     #do the forward part of the network
#     (O1,maxI1)=forward(model.first,ds.x,ds.subbags);
#     (O2,maxI2)=forward(model.second,O1,ds.bags); 
#     O3=squeeze(forward(model.third,O2),1);

#     idxs=sortperm(O3,rev=true);
#     idxs=idxs[O3[idxs].>onlypositive];

#     if onlynegative
#         idxs=sortperm(O3);
#         idxs=idxs[O3[idxs].<0];
#     end

#     if nreturned>0
#         idxs=idxs[1:min(nreturned,length(idxs))];
#     end

#     explanations=[];
#     for idx in idxs
#         (W2,B2,idxmap2)=explainlayer(maxI2[:,idx],model.second,squeeze(model.third.W,2));
#         OO=zeros(0,4);
#         for ii in keys(idxmap2)
#             (W1,B1,idxmap1)=explainlayer(maxI1[:,ii],model.first,W2[:,idxmap2[ii]]);
#             for xi in keys(idxmap1)
#                 contrib=dot(ds.x[:,xi],W1[:,idxmap1[xi]])+B1[idxmap1[xi]]+B2[idxmap2[ii]];
#     #             @printf("%d %d %d: %f\n",idx,ii,xi,contrib)
#                 OO=vcat(OO,transpose([idx,ii,xi,contrib]));
#             end
#         end
#         I=sortperm(OO[:,4],rev=true);
#         OO=OO[I,:];
#         mask=OO[:,4].>0;
#         OO=OO[mask,:]
#         push!(explanations,OO)
#     end
#     return(explanations,O3)
# end


"""

function explain(ds::DoubleBagModel,ds::DoubleBagDataset;threshold::Float32=0)

This function selects all samples for which the DoubleBagModel outputs a value greater than threshold,
and identifies the most contributing subbags causing them to be positive.
The identification is done iteratively by greedily removing subbags contributing most to the positiveness of the sample.

The function returns array of triplets, each describing one user:5min

First is the index of the bag (sample) that is being explaint
Second is the array of indexes of subbags that are causing the samples to be positive 
Third is the array of outputs on subbags, the length of this array is the same as the length of the second column
"""
function explain(model::DoubleBagModel,ds::DoubleBagDataset;threshold=0)
    #explain only samples above the threshold (default 0)
    O=forward!(model,ds);
    results=Array{Tuple{Int64,Array{Int64,1},Array{Float32,1}},1}(0)
    for j in find(O.>threshold)
        # create a small dataset containing only the given sample and initiates arrays 
        # for selected subbags and their outputs
        dss=ds[j:j]
        allbags=dss.bags[1]
        selected=Array{Int,1}(0)
        Os=Array{Float32,1}(0)
        active=collect(1:length(allbags));

        #iterate over active subbags
        while length(active)>0
          active=setdiff(active,selected)
          mi=0;
          mo=typemin(Float32)

          #get output on all active subbags
          dss.bags[1]=allbags[active]
          fullO=forward!(model,dss)[1]
          if fullO<zero(Float32)
            break
          end

          if length(active)==1 &&  fullO>0 
            push!(selected,active[1]);
            push!(Os,fullO);
            break;
          end

          #remove subbags one by one and pick the one with the highest score
          for i in active
            dss.bags[1]=allbags[setdiff(active,i)]
            O=forward!(model,dss)[1]
            if mo<fullO-O
              mo=fullO-O;
              mi=i;
            end
          end

          push!(selected,mi);
          push!(Os,mo);
        end
        push!(results,(j,ds.bags[j][selected],Os))
    end
    return(results)
end

# function ReluMaxReluMaxModel(k::Tuple{Int,Int,Int,Int},loss::EduNets.AbstractLoss;T::DataType=Float64)
#   return(DoubleBagModel(VoidLayer(k[1];T=T),
#     ReluMaxLayer((k[1],k[2]);T=T),
#     VoidLayer(k[2];T=T),
#     ReluMaxLayer((k[2],k[3]);T=T),
#     LinearLayer((k[3],k[4]);T=T),
#     loss,T=T));
# end

# function ReluMeanReluMeanModel(k::Tuple{Int,Int,Int,Int},loss::EduNets.AbstractLoss;T::DataType=Float64)
#   return(DoubleBagModel(VoidLayer(k[1];T=T),
#     ReluMeanLayer((k[1],k[2]);T=T),
#     VoidLayer(k[2];T=T),
#     ReluMeanLayer((k[2],k[3]);T=T),
#     LinearLayer((k[3],k[4]);T=T),
#     loss,T=T));
# end
