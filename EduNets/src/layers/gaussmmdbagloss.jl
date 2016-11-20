
export GaussMMDBagLoss;


"""
MMD loss with Gaussian kernel
"""
type GaussMMDBagLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
  D::Matrix{T};   #preallocated space for distance matrices
  gamma::T        #default value of gamma parameter used if bag has size one, default is one
end

function GaussMMDBagLoss(;T::DataType=Float64)
  return(GaussMMDBagLoss(zeros(T,0,0),zeros(T,0,0),one(T)));
end

function GaussMMDBagLoss(gamma;T::DataType=Float64)
  return(GaussMMDBagLoss(zeros(T,0,0),zeros(T,0,0),T(gamma)));
end

function forward!{T}(loss::GaussMMDBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  #check that we have enough large buffer
  l=max(maximum(map(length,xbags)),maximum(map(length,ybags)))
  if size(loss.D,2)<l
    loss.D=zeros(T,l,l);
  end

  #proceed with calculating the loss function
  f=zero(T);
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]


    dyy=view(loss.D,1:length(ybag),1:length(ybag))
    pairwise!(dyy,SqEuclidean(),view(Y,:,ybag))
    gamma=(size(dyy,2)>1)?1/median(dyy[[i>j for i in 1:size(dyy,2),j in 1:size(dyy,2)]]):loss.gamma;
    f+=mean(exp(-gamma*dyy)) 
    
    dxx=view(loss.D,1:length(xbag),1:length(xbag))
    pairwise!(dxx,SqEuclidean(),view(X,:,xbag))
    f+=mean(exp(-gamma*dxx))  

    dxy=view(loss.D,1:length(xbag),1:length(ybag))
    pairwise!(dxy,SqEuclidean(),view(X,:,xbag),view(Y,:,ybag))
    f-=2*mean(exp(-gamma*dxy))  
  end
  l=length(xbags);
  return(f/l);
end


function gradient!{T}(loss::GaussMMDBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  checksize!(X,loss);

  #check that we have enough large buffer
  l=max(maximum(map(length,xbags)),maximum(map(length,ybags)))
  if size(loss.D,2)<l
    loss.D=zeros(T,l,l);
  end

  #proceed with calculating the loss function
  f=zero(T);
  l=length(xbags);
  fill!(loss.gX,0)

  for (bind,xbag) in enumerate(xbags)
    ybag=ybags[bind]

    #this does not affect the final gradient, but it is better for the consistency
    dyy=view(loss.D,1:length(ybag),1:length(ybag))
    pairwise!(dyy,SqEuclidean(),view(Y,:,ybag))
    gamma=(size(dyy,2)>1)?1/median(dyy[[i>j for i in 1:size(dyy,2),j in 1:size(dyy,2)]]):loss.gamma;
    f+=mapreduce(s->exp(-gamma*s),+,dyy)/(l*length(dyy))
    
    dxx=view(loss.D,1:length(xbag),1:length(xbag))
    pairwise!(dxx,SqEuclidean(),view(X,:,xbag))
    for (iind,i) in enumerate(xbag)
      for jind in 1:iind-1
        j=xbag[jind]
        ff=exp(-gamma*dxx[iind,jind])/(l*length(dxx))
        loss.gX[:,j]+=4*gamma*ff*(X[:,i]-X[:,j])
        loss.gX[:,i]+=4*gamma*ff*(X[:,j]-X[:,i])
        f+=2*ff
      end
      f+=1/(l*length(dxx))
    end

    dxy=view(loss.D,1:length(xbag),1:length(ybag))
    pairwise!(dxy,SqEuclidean(),view(X,:,xbag),view(Y,:,ybag))
    for (iind,i) in enumerate(xbag)
      for (jind,j) in enumerate(ybag)
        ff=exp(-gamma*dxy[iind,jind])/(l*length(dxy))
        loss.gX[:,i]-=4*gamma*ff*(Y[:,j]-X[:,i])
        f-=2*ff
      end
    end
  end
  l=length(xbags);
  return(f,view(loss.gX,1:size(X,1),1:size(X,2)));
end

function checksize!{T}(X::AbstractArray{T},loss::GaussMMDBagLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end