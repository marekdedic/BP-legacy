using DataFrames
import Base: size,vcat,length,sub;
import Base.Operators.getindex;
import StatsBase: sample;


export SortedDoubleBagDataset,getindex,samplebags,createbagsandsubbags;

""" Bigbags --- range of instances of bags.
    bags --- ranges of subbags.
    subbags --- ranges of instances of subbags
    x --- data
    y --- labels
"""
type SortedDoubleBagDataset{T<:AbstractFloat}
  x::Matrix{T};
  y::Vector{Int};
  bigbags::Array{UnitRange{Int64},1}; 
  bags::Array{UnitRange{Int64},1};
  subbags::Array{UnitRange{Int64},1};

  info::DataFrames.DataFrame;
end

function findranges(ids)
  bags=fill(0:0,length(unique(ids)))
  idx=1
  bidx=1
  for i in 2:length(ids)
    if ids[i]!=ids[idx]
      bags[bidx]=idx:i-1
      idx=i;
      bidx+=1;
    end
  end
  if bidx<=length(bags)
    bags[bidx]=idx:length(ids)
  end
  return(bags)
end

function SortedDoubleBagDataset(bagids,subbagids,X::Matrix,labels::Vector{Int};info=DataFrames.DataFrame([]))
  #permute the datasets such that all ids ans subids are continuous
  I1=sortperm(subbagids);
  I2=sortperm(bagids[I1]);
  I=I1[I2]
  bagids=bagids[I];
  subbagids=subbagids[I];
  X=X[:,I]
  labels=labels[I]
  if !isempty(info)
    info=info[I,:]
  end

  #create the bags and subbags
  bigbags=findranges(bagids);
  subbags=Array{UnitRange{Int64},1}(0)
  bags=fill(0:0,length(bigbags))
  offset=0
  for (i,bag) in enumerate(bigbags)    
    sidx=length(subbags);  
    append!(subbags,map(r->offset+r,findranges(subbagids[bag])))
    bags[i]=sidx+1:length(subbags)
    offset=bag.stop
  end

  bagy=map(bag->maximum(labels[bag]),bigbags);
  return(SortedDoubleBagDataset(X,bagy,bigbags,bags,subbags,info))
end

function getindex(ds::SortedDoubleBagDataset,bagindexes::AbstractArray{Int})
  instanceidxs=mapreduce(i->collect(bigbags[i]),vcat,bagindexes)

  #allocate space for new bags and subbags
  bigbags=fill(0:0,length(bagindexes));
  bags=fill(0:0,length(bagindexes));
  subbags=fill(0:0,mapreduce(i->length(ds.bags[i]),+,bagindexes));

  subbagoffset=1
  instanceoffset=1
  for i in bagindexes
    bag[i]=subbagoffset:subbagoffset+length(ds.bags[i])-1
    subbagoffset+=length(ds.bags[i]);
    s=instanceoffset
    for j in ds.bags[i]
      subbags[j]=instanceoffset:instanceoffset+length(ds.subbags[j])-1
      instanceoffset+=length(ds.subbags[j])
    end
    bigbags[i]=s:instanceoffset-1
  end

  if size(ds.info,1)<maximum(instanceidxs)
    return(SortedDoubleBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bigbags,bags,subbags,DataFrames.DataFrame([])))
  else
    return(SortedDoubleBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bigbags,bags,subbags,ds.info[instanceidxs,:]))
  end
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

"""function getindex(ds::SortedDoubleBagDataset,bagindexes::AbstractArray{Int};subbagsize::Int=0)
returns a subset into the ds selecting samples (bags) with indexes bagindexes.
X and Y are allocated as new arrays
if subbagsize are greater than zero, then each subbag has maximum of subbagsize items (sampled randomly) """
function getindex(ds::SortedDoubleBagDataset,bagindexes::AbstractArray{Int};subbagsize::Int=0)
  (bags,subbagidxs)=remap(ds.bags[bagindexes]);

  subbags=deepcopy(ds.subbags[subbagidxs]);
  if subbagsize>0
    subbags=map(idxs->(length(idxs)>subbagsize)?sample(idxs,subbagsize,replace=false):idxs,subbags)
  end
  (subbags,instanceidxs)=remap(subbags)
  if size(ds.info,1)<maximum(instanceidxs)
    return(SortedDoubleBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bags,subbags,DataFrames.DataFrame([])))
  else
    return(SortedDoubleBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bags,subbags,ds.info[instanceidxs,:]))
  end
end

"""function view(ds::SortedDoubleBagDataset,bagindexes::AbstractArray{Int};subbagsize::Int=0)
returns a view into the ds selecting samples (bags) with indexes bagindexes.
X and Y are views into the dataset.
if subbagsize are greater than zero, then each subbag has maximum of subbagsize items (sampled randomly) """
function view(ds::SortedDoubleBagDataset,bagindexes::AbstractArray{Int};subbagsize::Int=0)
  #find which sub-bags will be in the new subset
  bags=deepcopy(ds.bags[bagindexes])
  subbagidxs=remap!(bags);

  subbags=deepcopy(ds.subbags[subbagidxs]);
  if subbagsize>0
    subbags=map(idxs->(length(idxs)>subbagsize)?sample(idxs,subbagsize,replace=false):idxs,subbags)
  end
  instanceidxs=remap!(subbags)
  if size(ds.info,1)<maximum(instanceidxs)
    return(SortedDoubleBagDataset(view(ds.x,:,instanceidxs),view(ds.y,bagindexes),bags,subbags,DataFrames.DataFrame([])))
  else
    return(SortedDoubleBagDataset(view(ds.x,:,instanceidxs),view(ds.y,bagindexes),bags,subbags,ds.info[instanceidxs,:]))
  end
end

function samplebags(ds::SortedDoubleBagDataset,n::Int64;subbagsize::Int=0)
  indexes=sample(1:length(ds.bags),n,replace=false);
  return(view(ds,indexes;subbagsize=subbagsize));
end

function samplebags(ds::SortedDoubleBagDataset,n::Array{Int64};subbagsize::Int=0)
  classbagids=map(i->findn(ds.y.==i),1:maximum(ds.y));
  indexes=mapreduce(i->sample(classbagids[i],minimum([length(classbagids[i]),n[i]]);replace=true),append!,1:length(n));
  return(view(ds,indexes;subbagsize=subbagsize));
end

function sample(ds::SortedDoubleBagDataset,n::Int64;subbagsize::Int=0)
  indexes=sample(1:length(ds.bags),n,replace=false);
  return(getindex(ds,indexes;subbagsize=subbagsize));
end

function sample(ds::SortedDoubleBagDataset,n::Array{Int64};subbagsize::Int=0)
  classbagids=map(i->findn(ds.y.==i),1:maximum(ds.y));
  indexes=mapreduce(i->sample(classbagids[i],minimum([length(classbagids[i]),n[i]]);replace=true),append!,1:length(n));
  return(getindex(ds,indexes;subbagsize=subbagsize));
end

function vcat(d1::SortedDoubleBagDataset,d2::SortedDoubleBagDataset)
  #we need to redefine bags and sub-bags, of them needs to be shifted by the number of bags / instances in d1
  l=length(d1.subbags)
  bags=vcat(deepcopy(d1.bags),map!(i->i.+l,deepcopy(d2.bags),d2.bags));
  l=length(size(d1.x,2))
  subbags=vcat(deepcopy(d1.subbags),map!(i->i.+l,deepcopy(d2.subbags),d2.subbags));

  ds=SortedDoubleBagDataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),bags,subbags,vcat(d1.info,d2.info));
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
