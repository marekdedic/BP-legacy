
export KLDivYXBagLoss;


"""
MMD loss with Gaussian kernel
"""
type KLDivYXBagLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
  Dyy::Matrix{T};   #preallocated space for distance matrices
  Dxy::Matrix{T};   #preallocated space for distance matrices
  k::Int        #default value of gamma parameter used if bag has size one, default is one
  warn::Bool
end

"""
    KLDivYXBagLoss(k::Int;T::DataType=Float64)

    k the number of nearest neighbours to estimate the density at a given point
    if warn is true, functions returned the number of skipped samples due bag containing only one instace.
"""
function KLDivYXBagLoss(k::Int;T::DataType=Float64,warn::Bool=false)
  return(KLDivYXBagLoss(zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),k,warn));
end

"""
    KLDivYXBagLoss(T::DataType=Float64)

    The number of nearest neighbours is dynamically set as a floor(sqrt(length(bag))
    if warn is true, functions returned the number of skipped samples due bag containing only one instace.
"""
function KLDivYXBagLoss(;T::DataType=Float64,warn::Bool=false)
  return(KLDivYXBagLoss(zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),0,warn));
end

function forward!{T}(loss::KLDivYXBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  #check that we have enough large buffer
  l=max(maximum(map(length,xbags)),maximum(map(length,ybags)))
  if size(loss.Dxy,2)<l
    loss.Dyy=zeros(T,l,l);
    loss.Dxy=zeros(T,l,l);
  end

  #proceed with calculating the loss function
  f=zero(T);
  skipped=0
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]
    if loss.k==0
      k=Int(floor(sqrt(min(length(xbag),length(ybag)))))
    else
      k=loss.k;
    end
    k=min(k,length(xbag),length(ybag)-1)
    if k==0
      skipped+=1;
      continue
    end

    c=size(X,1)/length(ybag)

    #calculate distances from reference distribution to itself
    dyy=view(loss.Dyy,1:length(ybag),1:length(ybag))
    pairwise!(dyy,SqEuclidean(),view(Y,:,ybag))
    
    #calculate distances from reference distribution to optimized
    dyx=view(loss.Dxy,1:length(ybag),1:length(xbag))
    pairwise!(dyx,SqEuclidean(),view(Y,:,ybag),view(X,:,xbag))

    #update the loss function
    for j in 1:length(ybag)
      sy=select!(view(dyy,j,:),k+1)
      sx=select!(view(dyx,j,:),k)
      f+=c*log(sy/sx);
    end
  end
  if loss.warn && skipped>0
    println("$skipped bags skipped because of zero length")
  end
  l=length(xbags);
  return(f/l);
end


function gradient!{T}(loss::KLDivYXBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  checksize!(X,loss);
  fill!(loss.gX,0)
  #check that we have enough large buffer
  l=max(maximum(map(length,xbags)),maximum(map(length,ybags)))
  if size(loss.Dxy,2)<l
    loss.Dyy=zeros(T,l,l);
    loss.Dxy=zeros(T,l,l);
  end

  #proceed with calculating the loss function
  f=zero(T);
  skipped=0
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]
    if loss.k==0
      k=Int(floor(sqrt(min(length(xbag),length(ybag)))))
    else
      k=loss.k;
    end
    k=min(k,length(xbag),length(ybag)-1)
    if k==0
      skipped+=1;
      continue
    end

    
    c=size(X,1)/(length(xbags)*length(ybag))

    #calculate distances from reference distribution to itself
    dyy=view(loss.Dyy,1:length(ybag),1:length(ybag))
    pairwise!(dyy,SqEuclidean(),view(Y,:,ybag))
    
    #calculate distances from reference distribution to optimized
    dyx=view(loss.Dxy,1:length(ybag),1:length(xbag))
    pairwise!(dyx,SqEuclidean(),view(Y,:,ybag),view(X,:,xbag))

    #update the loss function
    for j in eachindex(ybag)
      sy=select!(view(dyy,j,:),k+1)
      ii=sortperm(view(dyx,j,:))[k]
      sx=dyx[j,ii]
      f+=c*log(sy/sx);  
      loss.gX[:,xbag[ii]]-=2*c*(X[:,xbag[ii]]-Y[:,ybag[j]])/sx
    end
  end
  if loss.warn && skipped>0
    println("$skipped bags skipped because of zero length")
  end
  return(f,view(loss.gX,1:size(X,1),1:size(X,2)));
end

function checksize!{T}(X::AbstractArray{T},loss::KLDivYXBagLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end