export testgradient,gradient_estimate
function gradient_estimate(f,x;h=1e-6,progress::Bool=false)
  g=zeros(eltype(x),size(x));
  xd=deepcopy(x);
  for i in 1:length(x)
    xd[i]+=h/2;
    fr=f(xd);
    xd[i]-=h;
    fl=f(xd);
    xd[i]+=h/2;
    g[i]=(fr-fl)/h;
    if progress
      print(".")
    end
  end
  if progress
    println()
  end
  return(g)
end


""" function testgradient(optFun,theta) 
  optfun is the optimization function (f,g)=optFun(theta), theta is the point in which the gradient will be estimated."""
function testgradient(optFun::Function,theta::AbstractArray;verbose::Int=0,h=1e-6,progress::Bool=false)
  (f,g)=optFun(theta);

  function fVal(x)
    (f,_)=optFun(x)
    return(f)
  end
  gRef=gradient_estimate(fVal,theta;progress=progress,h=h);
  if verbose>1
    display(hcat(g,gRef))
  end
  if verbose >0
    println("Difference in estimated and analytic gradient $(maxabs(g-gRef))")
  end
  return(maxabs(g-gRef))
end