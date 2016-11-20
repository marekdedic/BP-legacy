
export forward,gradient,HausdorffBagLoss;

using Distances

"""
Loss for two sets of points calculated as a distance of their means
"""
type HausdorffBagLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function HausdorffBagLoss(;T::DataType=Float64)
  return(HausdorffBagLoss(zeros(T,0,0)));
end


function forward!{T}(loss::HausdorffBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end

  #proceed with calculating the loss function
  f=zero(T);
  l=length(xbags);
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]
    sx=view(X,:,xbag)
    sy=view(Y,:,ybag)
    D = pairwise(SqEuclidean(),sx,sy)
    daB = maximum(minimum(D,2))
    dbA = maximum(minimum(D,1))
    f+= max(daB, dbA)
  end
  return(f/l);
end


function gradient!{T}(loss::HausdorffBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  checksize!(X,loss);

  gX=view(loss.gX,1:size(X,1),1:size(X,2))
  fill!(gX,0)

  #proceed with calculating the loss function
  f=zero(T);
  l=length(xbags);
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]
    sx=view(X,:,xbag)
    sy=view(Y,:,ybag)
    D = pairwise(SqEuclidean(),sx,sy)
    (t,aBj)=findmin(D,2)
    (daB,aBi)=findmax(t)
    (t,bAi)=findmin(D,1)
    (dbA,bAj)=findmax(t)
    dbA = maximum(minimum(D,1))
    idx=(dbA>daB)?bAi[bAj]:aBj[aBi]
    (i,j)=ind2sub(size(D),idx)
    i=xbag[i]
    j=ybag[j]
    f+= max(daB, dbA)
    # println("$(max(daB, dbA)) $(sumabs2(X[:,i]-Y[:,j]))")
    gX[:,i]+=2*(X[:,i]-Y[:,j])/l
  end
  return(f/l,gX);
end

function checksize!{T}(X::AbstractArray{T},loss::HausdorffBagLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end