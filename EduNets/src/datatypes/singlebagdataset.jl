import Base: size,vcat,length,sub;
import Base.Operators.getindex;
import StatsBase: sample,nobs;

export SingleBagDataset,getindex,samplebags,simplybag,SingleBagModel;

"""In this dataset sub-bags is an array of arrays. Each array within an array defines indexes in X t, where each element of the master array corresponds to one subBag, 
and the array within holds indexes of instances belonging to the subBag.
 """
type SingleBagDataset{T<:AbstractFloat}<:AbstractDataset
  x::AbstractArray{T,2};
  y::AbstractArray{Int,1};
  bags::Bags;

  info::DataFrames.DataFrame;
end

function SingleBagDataset{T<:AbstractFloat}(x::AbstractArray{T,2},y::AbstractArray{Int,1},bags::Bags;info::DataFrames.DataFrame=DataFrames.DataFrame([]))
  return(SingleBagDataset(x,y,bags,info))
end


function SingleBagDataset{T<:AbstractFloat}(x::AbstractArray{T,2},y::AbstractArray{Int,1},bagids::AbstractArray;info::DataFrames.DataFrame=DataFrames.DataFrame([]))
  if length(y)==size(x,2)
    (bags,bagy)=simplybag(bagids,y);
    return(SingleBagDataset(x,bagy,bags,info))
  else
    (bags,_)=simplybag(bagids,zeros(Int,size(x,2)));
    return(SingleBagDataset(x,y,bags,info))
  end
end

function getobs(ds::SingleBagDataset,idxs)
  return(ds[idxs])
end

function nobs(ds::SingleBagDataset)
  return(length(ds.bags))
end



function getindex(ds::SingleBagDataset,bagindexes::AbstractArray{Int};bagsize::Int=0)
    #find which sub-bags will be in the new subset
  bags=deepcopy(ds.bags[bagindexes])
  if bagsize>0
    bags=map(idxs->(length(idxs)>bagsize)?sample(idxs,bagsize,replace=false):idxs,bags)
  end
  (bags,instanceidxs)=remap(bags);
  return(SingleBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bags,ds.info[instanceidxs,:]))
end


function sample(ds::SingleBagDataset,n::Int64;bagsize::Int=0)
  indexes=sample(1:length(ds.bags),n,replace=false);
  return(getindex(ds,indexes;bagsize=bagsize));
end

function sample(ds::SingleBagDataset,n::Array{Int64};bagsize::Int=0)
  classbagids=map(i->findn(ds.y.==i),1:maximum(ds.y));
  indexes=mapreduce(i->sample(classbagids[i],minimum([length(classbagids[i]),n[i]]);replace=true),append!,1:length(n));
  return(getindex(ds,indexes;bagsize=bagsize));
end

function vcat(d1::SingleBagDataset,d2::SingleBagDataset)
  #we need to redefine bags and sub-bags, of them needs to be shifted by the number of bags / instances in d1
  bags=deepcopy(d1.bags)
  for bag in d2.bags
    newbag=deepcopy(bag)+size(d1.x,2)
    push!(bags,newbag)
  end

  ds=SingleBagDataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),bags,vcat(d1.info,d2.info));
  return(ds)
end

function size(d::SingleBagDataset)
  return(length(d.bags),size(d.x,1))
end

function size(ds::SingleBagDataset,n::Int)
  if n==1
    return length(ds.bags)
  else 
    return size(ds.x,1)
  end
end

function simplybag(hosts,labels)
  #users are bagged based on the type-stamp when the connection has started and user's IP address
  bagmap=Dict{eltype(hosts),Array{Int64}}()
  for i in 1:length(hosts)
    key=hosts[i]
    if !haskey(bagmap,key)
      bagmap[key]=[i]
    else
      push!(bagmap[key],i)
    end
  end

  #once we have determined bags, we put them into the structure
  bags=Array{Array{Int64,1},1}(length(bagmap));
  bagindex=1;
  bagy=Array{Int64,1}(length(bagmap));
  for bag in bagmap
    y=maximum(labels[bag[2]])
    bags[bagindex]=bag[2];
    bagy[bagindex]=y;
    bagindex+=1
  end
  bagy=bagy;
  return(bags,bagy)
end