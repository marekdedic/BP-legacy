
export forward,gradient,MeanL2BagLoss;


"""
Loss for two sets of points calculated as a distance of their means
"""
type MeanL2BagLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
end

function MeanL2BagLoss(;T::DataType=Float64)
  return(MeanL2BagLoss(zeros(T,0,0)));
end


function forward!{T}(loss::MeanL2BagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end

  #proceed with calculating the loss function
  f=zero(T);
  l=length(xbags);
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]
    mx=mean(view(X,:,xbag),2)
    my=mean(view(Y,:,ybag),2)
    d=mx-my
    f+=sumabs2(d)
  end
  return(f/l);
end


function gradient!{T}(loss::MeanL2BagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  checksize!(X,loss);

  #proceed with calculating the loss function
  f=zero(T);
  l=length(xbags);
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]
    mx=mean(view(X,:,xbag),2)
    my=mean(view(Y,:,ybag),2)
    d=mx-my
    f+=sumabs2(d)
    for j in xbag
      loss.gX[:,j]=2*d/(length(xbag)*l)
    end
  end
  return(f/l,view(loss.gX,1:size(X,1),1:size(X,2)));
end

function checksize!{T}(X::AbstractArray{T},loss::MeanL2BagLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end