using DataFrames
import Base: size,vcat,getindex
import StatsBase: sample,nobs

export sample,vcat,Dataset,size,nobs,getobs

"""In this dataset bags and sub-bags indentify uniquely bags, which does not capture well the network scenarion, where one
    instance be part of multiple bags """
type Dataset{T<:AbstractFloat}<:AbstractDataset
  x::AbstractArray{T,2};
  y::AbstractArray{Int,1};

  info::DataFrame;
end

function Dataset{T}(x::AbstractArray{T,2},y::AbstractArray{Int,1})
  Dataset(x::AbstractArray{T,2},y::AbstractArray{Int,1},DataFrame(index=1:length(y)))
end

function size(ds::Dataset)
  return(size(ds.x,2),size(ds.x,1))
end

function getobs(ds::Dataset,idxs::UnitRange)
  return(Dataset(view(ds.x,:,idxs),view(ds.y,idxs)))
end

function getobs(ds::Dataset,idxs)
  return(Dataset(ds.x[:,idxs],ds.y[idxs]))
end

function nobs(ds::Dataset)
  return size(ds.x,2)
end

function size(ds::Dataset,n::Int)
  if n==1
    return size(ds.x,2)
  else 
    return size(ds.x,1)
  end
end

function getindex(ds::Dataset,idxs)
  return(Dataset(ds.x[:,idxs],ds.y[idxs],ds.info[idxs,:]))
end

function sample(ds::Dataset,n::Int64)
  n=(n>length(ds.y))?length(ds.y):n;
  return(subset(ds,StatsBase.sample(1:length(ds.y),n,replace=false)))
end

function sample(ds::Dataset,n::Array{Int,1})
  classids=map(i->find(ds.y.==i),1:maximum(ds.y))
  index=mapreduce(i->StatsBase.sample(classids[i],n[i],replace=false),vcat,1:maximum(ds.y))
  return(subset(ds,index))
end

"""returns subset of dataset set"""
function subset(ds::Dataset,indexes::AbstractArray{Int})
  return(Dataset(ds.x[:,indexes],ds.y[indexes]))
end

"""returns subset of dataset set"""
function subview(ds::Dataset,indexes::AbstractArray{Int})
  return(Dataset(view(ds.x,:,indexes),ds.y[indexes]))
end

function vcat(d1::Dataset,d2::Dataset)
  return(Dataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),vcat(d1.info,d2.info));)
end