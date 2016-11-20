

function uniqueIC(x::AbstractArray{Any,1})
  l=length(x)
  si=vcat(1,find(x[1:l-1].!=x[2:l])+1)
  lsi=length(si);
  iC=lsi*ones(Int64,l);
  for i in 1:lsi-1;
    iC[si[i]:si[i+1]-1]=i;
  end
  return(iC)
end

function uniqueIC(xx...)
  l=length(xx[1])
  mask=fill(false,l-1)
  # for i in 1:length(xx)
    # x=xx[i]
  for x in xx
    mask=mask | (x[1:l-1].!=x[2:l])
  end

  si=vcat(1,find(mask)+1)
  lsi=length(si);
  iC=lsi*ones(Int64,l);
  for i in 1:lsi-1;
    iC[si[i]:si[i+1]-1]=i;
  end
  return(iC)
end