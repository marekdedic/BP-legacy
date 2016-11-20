using DataFrames
import Base: size,vcat,length,sub;
import Base.Operators.getindex;
import StatsBase: sample,nobs;


export DoubleBagDataset,getindex,samplebags,createbagsandsubbags;

"""In this dataset sub-bags is an array of arrays. Each array within an array defines indexes in X t, where each element of the master array corresponds to one subBag, 
and the array within holds indexes of instances belonging to the subBag.
 """
type DoubleBagDataset{T<:AbstractFloat}<:AbstractDataset
  x::AbstractArray{T,2};
  y::AbstractArray{Int,1};
  bags::Array{Array{Int,1},1};
  subbags::Array{Array{Int,1},1};

  info::DataFrames.DataFrame;
end


function DoubleBagDataset(bagids,subbagids,X::Matrix,labels::Vector{Int};info=DataFrames.DataFrame([]))
  (bags,subbags,bagy)=createbagsandsubbags(bagids,subbagids,labels)
  return(DoubleBagDataset(X,bagy,bags,subbags,info))
end

function DoubleBagDataset(X::Matrix,labels::Vector{Int},bagids,subbagids;info=DataFrames.DataFrame([]))
  (bags,subbags,bagy)=createbagsandsubbags(bagids,subbagids,labels)
  return(DoubleBagDataset(X,bagy,bags,subbags,info))
end

function getobs(ds::DoubleBagDataset,idxs)
  return(ds[idxs])
end

function nobs(ds::DoubleBagDataset)
  return(length(ds.bags))
end

""" changes the indexes inside the bags to be linear, and return indexes of bags in the order 1,2,3,4,...  """
function remap(bags::Array{Array{Int,1}})
  mx=0;
  for idxs in bags
      if !isempty(idxs)
          mx=max(mx,maximum(idxs))
      end
  end
  presentlist=falses(mx)
  for idxs in bags
    presentlist[idxs]=true;
  end
  indexes=findn(presentlist)
  bagmap=Dict{Int64,Int64}(zip(indexes,1:length(indexes)));
  # map!(bag->map!(i->bagmap[i],bag,bag),bags,bags);
  newbags=Array{Array{Int,1,},1}(length(bags))
  for (i,bg) in enumerate(bags)
    newbags[i]=map(i->bagmap[i],bg)
  end
  return(newbags,indexes)
end

""" changes the indexes inside the bags to be linear, and return indexes of bags in the order 1,2,3,4,...  """
function remap(bags::Array{UnitRange{Int64}})
  newbags=Array{UnitRange{Int64}}(length(bags))

  idx=1
  l=mapreduce(length,+,bags)
  indexes=zeros(Int,l);
  for (i,bag) in enumerate(bags)
    newbags[i]=idx:idx+length(bag)-1
    indexes[newbags[i]]=collect(bag)
    idx+=length(bag)
  end
  return(newbags,indexes)
end

"""function getindex(ds::DoubleBagDataset,bagindexes::AbstractArray{Int};subbagsize::Int=0)
returns a subset into the ds selecting samples (bags) with indexes bagindexes.
X and Y are allocated as new arrays
if subbagsize are greater than zero, then each subbag has maximum of subbagsize items (sampled randomly) """
function getindex(ds::DoubleBagDataset,bagindexes::AbstractArray{Int};subbagsize::Int=0)
  (bags,subbagidxs)=remap(ds.bags[bagindexes]);

  subbags=deepcopy(ds.subbags[subbagidxs]);
  if subbagsize>0
    subbags=map(idxs->(length(idxs)>subbagsize)?sample(idxs,subbagsize,replace=false):idxs,subbags)
  end
  (subbags,instanceidxs)=remap(subbags)
  if size(ds.info,1)<maximum(instanceidxs)
    return(DoubleBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bags,subbags,DataFrames.DataFrame([])))
  else
    return(DoubleBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bags,subbags,ds.info[instanceidxs,:]))
  end
end

function sample(ds::DoubleBagDataset,n::Int64;subbagsize::Int=0)
  indexes=sample(1:length(ds.bags),n,replace=false);
  return(getindex(ds,indexes;subbagsize=subbagsize));
end

function sample(ds::DoubleBagDataset,n::Array{Int64};subbagsize::Int=0)
  classbagids=map(i->findn(ds.y.==i),1:maximum(ds.y));
  indexes=mapreduce(i->sample(classbagids[i],minimum([length(classbagids[i]),n[i]]);replace=true),append!,1:length(n));
  return(getindex(ds,indexes;subbagsize=subbagsize));
end

function vcat(d1::DoubleBagDataset,d2::DoubleBagDataset)
  #we need to redefine bags and sub-bags, of them needs to be shifted by the number of bags / instances in d1
  l=length(d1.subbags)
  bags=vcat(deepcopy(d1.bags),map!(i->i.+l,deepcopy(d2.bags),d2.bags));
  l=length(size(d1.x,2))
  subbags=vcat(deepcopy(d1.subbags),map!(i->i.+l,deepcopy(d2.subbags),d2.subbags));

  ds=DoubleBagDataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),bags,subbags,vcat(d1.info,d2.info));
  return(ds)
end

function createbagsandsubbags(bagids,subbagids,labels)
  bag2instance=Dict{eltype(bagids),Array{Int64}}();
  for i in 1:length(bagids)
    key=bagids[i]
    if !haskey(bag2instance,key)
      bag2instance[key]=[i]
    else
      push!(bag2instance[key],i)
    end
  end

  #once we have indexes of instances within each bag, we need to determine indexes of subbags
  subbags=Array{Array{Int64,1},1}(0);
  bags=Array{Array{Int64,1},1}(length(bag2instance));
  bagindex=1;
  bagy=Array{Int64,1}(length(bag2instance));
  for bag in bag2instance
    y=0;
    bagsofbag=Dict{eltype(subbagids),Array{Int64}}()
    #iterate over indexes and create bags groupping connections with the same source / destination IP address
    for i in bag[2]
      host=subbagids[i]
      if !haskey(bagsofbag,host)
        bagsofbag[host]=[i]
      else
        push!(bagsofbag[host],i)
      end
      y=(y>labels[i])?y:labels[i];
    end

    #add the bag and its subbags to corresponding arrays 
    bags[bagindex]=length(subbags)+collect(1:length(bagsofbag));
    bagy[bagindex]=y;
    append!(subbags,collect(values(bagsofbag)));
    bagindex+=1;
  end
  return(bags,subbags,bagy)
end
