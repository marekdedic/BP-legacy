
export KLDivXYBagLoss;


"""
MMD loss with Gaussian kernel
"""
type KLDivXYBagLoss{T<:AbstractFloat}<:AbstractLoss
  gX::Matrix{T};   #preallocated space for returning gradient with respect to input
  Dxx::Matrix{T};   #preallocated space for distance matrices
  Dxy::Matrix{T};   #preallocated space for distance matrices
  k::Int        #default value of gamma parameter used if bag has size one, default is one
  warn::Bool
end

"""
    KLDivXYBagLoss(k::Int;T::DataType=Float64,warn::Bool=false)

    k the number of nearest neighbours to estimate the density at a given point
    if warn is true, functions returned the number of skipped samples due bag containing only one instace. 
"""
function KLDivXYBagLoss(k::Int;T::DataType=Float64,warn::Bool=false)
  return(KLDivXYBagLoss(zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),k,warn));
end


"""
    KLDivXYBagLoss(;T::DataType=Float64,warn::Bool=false)

    The number of nearest neighbours is dynamically set as a floor(sqrt(length(bag))
    if warn is true, functions returned the number of skipped samples due bag containing only one instace. 
"""
function KLDivXYBagLoss(;T::DataType=Float64,warn::Bool=false)
  return(KLDivXYBagLoss(zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),0,warn));
end



function forward!{T}(loss::KLDivXYBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  #check that we have enough large buffer
  l=max(maximum(map(length,xbags)),maximum(map(length,ybags)))
  if size(loss.Dxy,2)<l
    loss.Dxx=zeros(T,l,l);
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
    k=min(k,length(xbag)-1,length(ybag))
    if k==0
      skipped+=1;
      continue;
    end
    c=size(X,1)/length(xbag)


    #calculate distances from reference distribution to itself
    dxx=view(loss.Dxx,1:length(xbag),1:length(xbag))
    pairwise!(dxx,SqEuclidean(),view(X,:,xbag))
    
    #calculate distances from reference distribution to optimized
    dxy=view(loss.Dxy,1:length(xbag),1:length(ybag))
    pairwise!(dxy,SqEuclidean(),view(X,:,xbag),view(Y,:,ybag))

    #update the loss function
    for j in 1:length(xbag)
      sx=select!(view(dxx,j,:),k+1)
      sy=select!(view(dxy,j,:),k)
      f+=c*log(sx/sy);
    end
  end
  if loss.warn && skipped>0
    println("$skipped bags skipped because of zero length")
  end
  l=length(xbags);
  return(f/l);
end


function gradient!{T}(loss::KLDivXYBagLoss{T},X::StridedMatrix{T},xbags,Y::StridedMatrix{T},ybags)
  if length(xbags)!=length(ybags)
    error("Length of xbags and ybags should be the same")
  end
  checksize!(X,loss);
  fill!(loss.gX,0)
  #check that we have enough large buffer
  l=max(maximum(map(length,xbags)),maximum(map(length,ybags)))
  if size(loss.Dxy,2)<l
    loss.Dxx=zeros(T,l,l);
    loss.Dxy=zeros(T,l,l);
  end

  #proceed with calculating the loss function
  f=zero(T);
  skipped=0;
  for (i,xbag) in enumerate(xbags)
    ybag=ybags[i]
    if loss.k==0
      k=Int(floor(sqrt(min(length(xbag),length(ybag)))))
    else
      k=loss.k;
    end
    k=min(k,length(xbag)-1,length(ybag))
    if k==0
      skipped+=1;
      continue
    end
    c=size(X,1)/length(xbag)

    #calculate distances from reference distribution to itself
    dxx=view(loss.Dxx,1:length(xbag),1:length(xbag))
    pairwise!(dxx,SqEuclidean(),view(X,:,xbag))
    
    #calculate distances from reference distribution to optimized
    dxy=view(loss.Dxy,1:length(xbag),1:length(ybag))
    pairwise!(dxy,SqEuclidean(),view(X,:,xbag),view(Y,:,ybag))

    #update the loss function
    for j in eachindex(xbag)
      ix=sortperm(view(dxx,j,:))[k+1]
      iy=sortperm(view(dxy,j,:))[k]
      sy=dxy[j,iy]
      sx=dxx[j,ix]
      f+=c*log(sx/sy);  
      loss.gX[:,xbag[j]]+=2*c*(X[:,xbag[j]]-X[:,xbag[ix]])/sx
      loss.gX[:,xbag[ix]]+=2*c*(X[:,xbag[ix]]-X[:,xbag[j]])/sx
      loss.gX[:,xbag[j]]-=2*c*(X[:,xbag[j]]-Y[:,ybag[iy]])/sy
    end
  end
  if loss.warn && skipped>0
    println("$skipped bags skipped because of zero length")
  end
  return(f,view(loss.gX,1:size(X,1),1:size(X,2)));
end

function checksize!{T}(X::AbstractArray{T},loss::KLDivXYBagLoss{T})
  if length(loss.gX)<length(X)
    loss.gX=zeros(eltype(X),size(X))
  end
end