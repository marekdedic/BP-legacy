
export forward,gradient,PairwiseL2BagLoss;


"""
  PairwiseL2BagLoss{T<:AbstractFloat}<:AbstractLoss

  Loss for two sets of points calculated as a mean of all pairwise distances

"""
type PairwiseL2BagLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function PairwiseL2BagLoss(;T::DataType=Float64)
  return(PairwiseL2BagLoss(zeros(T,0,0)));
end


function forward!{T}(loss::PairwiseL2BagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end

  #proceed with calculating the loss function
  f=zero(T);
  l=length(xbags);
  for (xbag,ybag) in zip(xbags,ybags)
    d=pairwise(SqEuclidean(),X[:,xbag],Y[:,ybag])
    d=pairwise(SqEuclidean(),X[:,xbag],Y[:,ybag])
    f+=mean(d)
  end
  return(f/l);
end


function gradient!{T}(loss::PairwiseL2BagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  checksize!(X,loss);

  #proceed with calculating the loss function
  f=zero(T);
  l=length(xbags);
  fill!(loss.gX,0)
  for (xbag,ybag) in zip(xbags,ybags)
    c=1.0/(length(xbag)*length(ybag)*l);
    for i in xbag
      for j in ybag
        d=view(X,:,i)-view(Y,:,j)
        loss.gX[:,i]+=2*d*c
        f+=sumabs2(d)*c
      end
    end
  end
  return(f,view(loss.gX,1:size(X,1),1:size(X,2)));
end

function checksize!{T}(X::AbstractArray{T},loss::PairwiseL2BagLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end