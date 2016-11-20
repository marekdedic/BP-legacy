
using Distances

import Base.length
export forward,gradient,ClusterLoss;
# using ParallelAccelerator


type ClusterLoss{T<:AbstractFloat}<:AbstractLoss
  k::Int; #number of nearest neighbours we are going to use
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
  C::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function ClusterLoss(k::Int;T::DataType=Float64)
  return(ClusterLoss(k,zeros(T,0,0),zeros(T,0,0)));
end

function length(loss::ClusterLoss)
  return(0)
end

function filterindexes(loss::ClusterLoss,R,i,idxs)
  if loss.k>=length(idxs)
    return(idxs)
  end
  select!(idxs,loss.k,lt=(r,s)->R[i,r]<R[i,s])
  return(idxs[1:loss.k])
end

function forward{T}(loss::ClusterLoss{T},X::StridedMatrix{T},Y::StridedVector{Int})
  R=pairwise(SqEuclidean(),X)

  f=T(0)
  # @acc begin
    sameidxs=[filterindexes(loss,R,i,setdiff(find(Y.==Y[i]),i)) for i in 1:size(X,2)]
    otheridxs=[filterindexes(loss,R,i,setdiff(find(Y.!=Y[i]),i)) for i in 1:size(X,2)]
  # end
  for i in 1:size(X,2)
    #get indexes of instances closest to the given sample 
    for j in sameidxs[i]
      for k in otheridxs[i]
        f+=max(0,1+R[i,j]-R[i,k])
      end
    end
  end
  return(f)
end


function gradient!{T}(loss::ClusterLoss{T},X::StridedMatrix{T},Y::StridedVector{Int})
  checksize!(X,loss);
  R=pairwise(SqEuclidean(),X)

  f=T(0)
  fill!(loss.gX,0)
  fill!(loss.C,0)

  # @acc begin
  #   sameidxs=map(i->filterindexes(loss,R,i,setdiff(find(Y.==Y[i]),i)),1:size(X,2))
  #   otheridxs=map(i->filterindexes(loss,R,i,setdiff(find(Y.!=Y[i]),i)),1:size(X,2))
  # end
  sameidxs=[filterindexes(loss,R,i,setdiff(find(Y.==Y[i]),i)) for i in 1:size(X,2)]
  otheridxs=[filterindexes(loss,R,i,setdiff(find(Y.!=Y[i]),i)) for i in 1:size(X,2)]
  #get indexes of instances closest to the given sample 
  for i in 1:size(X,2)
    for j in sameidxs[i]
      for k in otheridxs[i]
        fs=1+R[i,j]-R[i,k]
        if fs>0
          loss.C[j,i]-=2;
          loss.C[j,j]+=2;
          loss.C[k,i]+=2;
          loss.C[k,k]-=2;
          loss.C[i,k]+=2;
          loss.C[i,j]-=2;
          f+=fs
        end
      end
    end
  end
  
  for i in 1:size(X,2)
    @simd for j in 1:size(X,2)
      @inbounds loss.gX[:,i]+=loss.C[i,j]*X[:,j]
    end
  end

  # print("diff ")
  # println(maxabs(X*loss.C'-view(loss.gX,:,1:length(Y))))
  # println("gradient of loss $(sumabs(loss.gX))")
  return(f,view(loss.gX,:,1:length(Y)))
end


# function gradient!{T}(loss::ClusterLoss{T},X::StridedMatrix{T,2},Y)
#   checksize!(X,loss);
#   R=pairwise(SqEuclidean(),X)

#   f=T(0)
#   fill!(loss.gX,0)

#   #get indexes of instances closest to the given sample 
#   sameidxs=map(i->filterindexes(loss,R,i,setdiff(find(Y.==Y[i]),i)),1:size(X,2))
#   otheridxs=map(i->filterindexes(loss,R,i,setdiff(find(Y.!=Y[i]),i)),1:size(X,2))
#   for i in 1:size(X,2)
#     for j in sameidxs[i]
#       for k in otheridxs[j]
#         fs=max(0,1+R[i,j]-R[i,k])
#         if fs>0
#           loss.gX[:,j]-=2*(view(X,:,i)-view(X,:,j))
#           loss.gX[:,k]+=2*(view(X,:,i)-view(X,:,k))
#           loss.gX[:,i]+=2*(view(X,:,k)-view(X,:,j))
#           f+=fs
#         end
#       end
#     end
#   end
#   return(f,view(loss.gX,:,1:length(Y)))
# end


function checksize!{T}(X::StridedMatrix{T},loss::ClusterLoss{T})
  if size(loss.gX,1)<size(X,1)||size(loss.gX,2)<size(X,2)
    loss.gX=zeros(eltype(X),size(X))
    loss.C=zeros(eltype(X),size(X,2),size(X,2))
  end
end