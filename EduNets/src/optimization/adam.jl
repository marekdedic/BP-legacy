using StatsBase
using JLD
export AdamOptions,adam,paralleladam,aadam;

"""AdamOption controls the convergence parameters of the ADAM algorithm: 
  maxIter=1e6;
  miniBatchSize=100;
  progressFile=""
  trackGF=false;
  verbose=100;
  alpha=0.001;
  beta1=0.99;
  beta2=0.999;
  epsilon=1e-8;"""
type AdamOptions{T<:AbstractFloat}
  maxIter::Int
  miniBatchSize::Int
  verbose::Int
  progressFile::AbstractString
  trackGF::Bool
  alpha::T
  beta1::T
  beta2::T
  epsilon::T
  errortol::T;
  progresstol::T;
end

function AdamOptions(;maxIter=Int(1e6),miniBatchSize=100,verbose=100,progressFile="",trackGF=false,alpha=0.001,beta1=0.99,beta2=0.999,epsilon=1e-8,errortol=0.0,progresstol=0.0)
  return AdamOptions{Float32}(maxIter,miniBatchSize,verbose,progressFile,trackGF,alpha,beta1,beta2,epsilon,errortol,progresstol);
end

"""The plain implementation of ADAM algorithm, (f,g)=fOpt(theta) is the optimized function, theta is the initial solution, and options are AdamOptions"""
function adam(fOpt,theta::AbstractArray,options::AdamOptions)
  m=zeros(eltype(theta),size(theta));
  v=zeros(eltype(theta),size(theta));
  is=1:options.maxIter
  (lastF,theta,_,_)=adam(fOpt,theta,m,v,is,options)
  return(theta);
end


"""Warm-start version of adam, where values of variables recording the state m v and iterations are provided""" 
function adam(fOpt,theta,m,v,is,options)
  sumF=0;
  cumTime=0;
  tic();
  lastF=0;
  for i in is
    #generate indexes of random samples
    (f,gf)=fOpt(theta);
    sumF=sumF+f;

    #make a step in Adam
    m=options.beta1*m + (1-options.beta1)*gf;
    v=options.beta2*v + (1-options.beta2)*gf.^2;
    mHat=m/(1-options.beta1^i);
    vHat=v/(1-options.beta2^i);
    theta=theta-options.alpha*mHat./(sqrt(v)+options.epsilon);
    if options.verbose>0 && mod(i,options.verbose)==0
      tt=toq();
      tic()
      cumTime+=tt;
      if abs(sumF/options.verbose - lastF) < options.progresstol
        break;
      end

      lastF=sumF/options.verbose
      println("$i error = $(lastF) time = $(cumTime) ($(tt))")
      sumF=0;
      if ~isempty(options.progressFile)
        save(options.progressFile,"theta",theta);
      end
      if lastF<options.errortol
        break;
      end
    end
  end
  toq()
  return(lastF,theta,m,v)
end

"""This function should run in parallel loading of the data from files in each parallel iteration"""
function adam{T}(loaddata::Function,optFun::Function,theta::Vector{T};options=EduNets.AdamOptions(),numberofdataloads=10000,errorfile="")
  #initialize the state
  m=zeros(eltype(theta),length(theta))
  v=zeros(eltype(theta),length(theta))
  ofprefix=options.progressFile;
  options.progressFile="";

  errors=zeros(eltype(theta),numberofdataloads)
  startiter=1;
  if !isempty(ofprefix) && isfile(@sprintf("%s.jld",ofprefix))
    println("loading the old state")
    startiter=load(@sprintf("%s.jld",ofprefix),"iter");
    theta=load(@sprintf("%s.jld",ofprefix),"theta");
    v=load(@sprintf("%s.jld",ofprefix),"v");
    m=load(@sprintf("%s.jld",ofprefix),"m");
  end

  for iter in startiter:numberofdataloads
    ds=loaddata();

    is=(iter-1)*options.maxIter+1:(iter*options.maxIter);
    (f,theta,m,v)=EduNets.adam(x->optFun(ds,x),theta,m,v,is,options)

    errors[iter]=f;
    if !isempty(ofprefix)
      writedlm(@sprintf("%s.txt",ofprefix),theta);
      save(@sprintf("%s.jld",ofprefix),"iter",iter,"theta",theta,"m",m,"v",v);
    end
  end
  return(theta)
end

"""This function should run in parallel loading of the data from files in each parallel iteration"""
function paralleladam{T}(loaddata::Function,optFun::Function,theta::AbstractArray{T};options=EduNets.AdamOptions(),numberofdataloads=10000,errorfile="")
  #initialize the state
  m=zeros(eltype(theta),length(theta))
  v=zeros(eltype(theta),length(theta))
  ofprefix=options.progressFile;
  options.progressFile="";
  errors=zeros(eltype(theta),numberofdataloads)

  startiter=1;
  if !isempty(ofprefix) && isfile(@sprintf("%s.jld",ofprefix))
    println("loading the old state")
    startiter=load(@sprintf("%s.jld",ofprefix),"iter");
    theta=load(@sprintf("%s.jld",ofprefix),"theta");
    v=load(@sprintf("%s.jld",ofprefix),"v");
    m=load(@sprintf("%s.jld",ofprefix),"m");
  end

  for iter in startiter:numberofdataloads
    @everywhere ds=loaddata();

    is=(iter-1)*options.maxIter+1:(iter*options.maxIter);
    (errors[iter],theta,m,v)=EduNets.adam(optFun,theta,m,v,is,options);

    if !isempty(errorfile)
      writedlm(errorfile,hcat(1:iter,errors[1:iter]));
    end

    if !isempty(ofprefix)
      writedlm(@sprintf("%s.txt",ofprefix),theta);
      save(@sprintf("%s.jld",ofprefix),"iter",iter,"theta",theta,"m",m,"v",v);
    end
  end
  return(theta)
end